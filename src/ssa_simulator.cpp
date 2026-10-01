// bngsim/src/ssa_simulator.cpp — SSA and PSA stochastic simulators
//
// SSA: Gillespie's direct method with dependency graph + 4-ary sum tree.
// PSA: Partial Scaling Algorithm (Lin, Feng, Hlavacek, J. Chem. Phys. 150, 244101, 2019).
//
// Optimizations:
//   1. Dependency graph: After reaction fires, only recompute propensities
//      for reactions whose propensity is affected by the changed species.
//      O(k) where k ≈ 5–20, instead of O(N) over all reactions.
//   2. 4-ary sum tree: O(log N) reaction selection
//      and O(log N) propensity updates, replacing O(N) linear scan.
//
// Per-instance RNG (std::mt19937_64). Deterministic seeding.
// Simulation time is always tracked so time() expressions stay current.
// (`time` is the only clock symbol; `t` is an ordinary model identifier.)

#include "bngsim/cc_jit.hpp"  // GH #190: system-cc propensity backend (no MIR)
#include "bngsim/mir_jit.hpp" // GH #149: opt-in JIT'd propensity fast path
#include "bngsim/model.hpp"
#include "bngsim/platform_compat.hpp" // POSIX ssize_t shim for Windows (GH #150)
#include "bngsim/result.hpp"
#include "bngsim/simulator.hpp"
#include "bngsim/types.hpp"
#include "bngsim/wallclock.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace bngsim {

// Out-of-line, rarely-taken paths kept out of the SSA loop's inlined code.
#if defined(_MSC_VER)
#define BNGSIM_SSA_COLD __declspec(noinline)
#else
#define BNGSIM_SSA_COLD __attribute__((noinline, cold))
#endif

// ─── Sum tree ────────────────────────────────────────────────────────────────
//
// Supports, for N non-negative weights:
//   - set(i, value): O(log N)
//   - total(): O(1)
//   - find(target): the index whose slice holds target, O(log N)
//
// Used for reaction selection: stores propensities (or scaled propensities
// for PSA), enabling O(log N) sampling instead of O(N) linear scan.
//
// Every internal node is the sum of its children, and an update recomputes
// each node on the leaf's path from them. It used to be a Fenwick tree, whose
// nodes are running sums that an update adjusts by a delta and nothing ever
// re-summed, so each update's rounding stayed in the node for the rest of the
// run (issue #713). A propensity of 1e11 that shared a node with one of 1e-6
// absorbed the small one's first write into its own rounding; when the large
// one decayed to 0 the node kept the residue, 1.1e-16, not 1e-6, and the slow
// channel never fired again, with no warning. A residue of the other sign made
// the total negative, which the loop read as "stuck". Here a node is a fresh
// sum of its current children, so once a large weight is gone, nothing of it
// is left behind.
//
// A leaf's slice is [sum of the weights before it, that sum + its weight). A
// target in [0, total()) therefore never selects a leaf of weight 0: the old
// find took the first index whose running sum reached the target, which for
// target 0 was index 0 whatever its weight.
//
// Why four children: an update's cost is a chain of dependent adds, one per
// level, and the next step reads the root straight away. A binary tree put
// 7 adds on that chain at 97 reactions and cost up to 21% of a step there;
// four children halve the depth, which brings the SSA back to the Fenwick
// tree's speed (0.99x-1.05x of it on the committed SSA suite's small
// networks, 0.8x-0.9x on the large ones).

class SumTree {
  public:
    SumTree() = default;

    // A 4-ary tree: node p's children are 4p-2 .. 4p+1 (the root is 1, its
    // children 2..5), and the leaves are the last level. Four children rather
    // than two halve the depth, which is what an update pays for (see set).
    explicit SumTree(int n) : n_(n) {
        int leaves = 1;
        while (leaves < n) {
            leaves *= 4;
            ++depth_;
        }
        base_ = (leaves - 1) / 3 + 1; // 1-based index of leaf 0
        // Offset by 2 so that every group of four siblings (4p-2 .. 4p+1)
        // starts on a multiple of four doubles.
        node_.assign(static_cast<std::size_t>(base_ + leaves) + 2, 0.0);
    }

    // Set leaf i to value and recompute its ancestors from their children. At
    // each level the sum of the three siblings does not depend on the new
    // value, so only one add per level is on the dependency chain; the running
    // sum is carried in a register. An unchanged value leaves every ancestor as
    // it is: a firing re-sets every reaction that reads a function it moved,
    // and most of those are unchanged.
    void set(int i, double value) {
        int j = base_ + i;
        if (at(j) == value)
            return;
        double v = value;
        at(j) = v;
        while (j > 1) {
            const int w = (j - 2) & 3; // position among its siblings
            const int c0 = j - w;
            v += at(c0 + (w ^ 1)) + (at(c0 + (w ^ 2)) + at(c0 + (w ^ 3)));
            j = (j + 2) >> 2;
            at(j) = v;
        }
    }

    double total() const { return at(1); }

    // The leaf whose slice holds target, for 0 <= target < total().
    int find(double target) const {
        int j = 1;
        for (int d = 0; d < depth_; ++d) {
            const int c0 = 4 * j - 2;
            const double a = at(c0);
            const double ab = a + at(c0 + 1);
            const double abc = ab + at(c0 + 2);
            if (target < a) {
                j = c0;
            } else if (target < ab) {
                target -= a;
                j = c0 + 1;
            } else if (target < abc) {
                target -= ab;
                j = c0 + 2;
            } else {
                target -= abc;
                j = c0 + 3;
            }
        }
        const int i = j - base_;
        if (i < n_ && at(j) > 0.0)
            return i;
        return nearest_positive(i);
    }

    double value(int i) const { return at(base_ + i); }

    int size() const { return n_; }

  private:
    double &at(int j) { return node_[static_cast<std::size_t>(j) + 2]; }
    double at(int j) const { return node_[static_cast<std::size_t>(j) + 2]; }

    // Rounding can carry a target that lies at the very end of the last
    // positive slice past it, into weight-0 leaves: find's prefix sums are not
    // added in the order set formed the node. It belongs to that last positive
    // leaf. Out of line: it is essentially never taken.
    BNGSIM_SSA_COLD int nearest_positive(int i) const {
        for (int k = std::min(i, n_ - 1); k >= 0; --k)
            if (at(base_ + k) > 0.0)
                return k;
        for (int k = i + 1; k < n_; ++k)
            if (at(base_ + k) > 0.0)
                return k;
        return std::min(i, n_ - 1);
    }

    int n_ = 0;
    int depth_ = 0;
    int base_ = 1;
    std::vector<double> node_;
};

// ─── Dependency Graph ────────────────────────────────────────────────────────
//
// Maps species → set of reactions whose propensity depends on that species.
//
// For Elementary rate laws: propensity depends on reactant species only.
// For MichaelisMenten rate laws: propensity depends on reactant species.
// For Functional rate laws: propensity depends on reactant species PLUS
//   all species that contribute to any observable (since function values
//   can depend on any observable via ExprTk expressions).
// Under PSA (psa_product_deps): every reaction ALSO depends on its product
//   species, because the leap factor iScaling is the min population over
//   reactants ∪ products (GH #14) — a change to a product can change the
//   reaction's scaled propensity. Exact SSA does not add this (products do
//   not enter the raw rate), keeping its recompute set minimal.
//
// When a reaction fires, it changes certain species (reactants consumed,
// products produced). We look up all reactions that depend on any changed
// species, giving us the minimal set of propensities to recompute.

struct DependencyGraph {
    // species_to_reactions[s] = sorted list of 0-based reaction indices
    // whose propensity depends on species s (0-based).
    std::vector<std::vector<int>> species_to_reactions;

    // reaction_to_affected_species[r] = sorted list of 0-based species indices
    // that change when reaction r fires (union of reactants + products, excluding fixed).
    std::vector<std::vector<int>> reaction_to_affected_species;

    // GH #190 — reaction_to_affected_reactions[r] = sorted, deduplicated list of
    // reactions whose propensity must be recomputed after reaction r fires. This
    // is a pure function of topology (the union over r's changed species of
    // species_to_reactions), so it is precomputed ONCE at build() rather than
    // re-derived with a per-step std::sort + std::unique + vector inserts. The
    // SSA hot loop then reads it as a const reference — zero per-step allocation.
    std::vector<std::vector<int>> reaction_to_affected_reactions;

    // has_functional_rates: true if any reaction uses Functional rate laws
    bool has_functional_rates = false;

    // Build dependency graph from model. psa_product_deps: also register each
    // reaction's dependency on its product species (needed only under PSA, where
    // iScaling is bounded by the min population over reactants ∪ products).
    void build(const NetworkModel &model, bool psa_product_deps) {
        const int ns = model.n_species();
        const int nr = model.n_reactions();
        const auto &reactions = model.reactions();
        const auto &species_list = model.species();
        const auto &observables = model.observables();

        has_functional_rates = false;
        species_to_reactions.assign(ns, {});
        reaction_to_affected_species.resize(nr);

        // 1. Determine which species are "observable species" —
        //    species that appear in ANY observable group.
        //    Functional rate laws' propensities depend on these indirectly.
        std::vector<bool> is_observable_species(ns, false);
        for (const auto &obs : observables) {
            for (const auto &entry : obs.entries) {
                int si = entry.species_index - 1; // 1-based → 0-based
                if (si >= 0 && si < ns) {
                    is_observable_species[si] = true;
                }
            }
        }

        // 2. For each reaction, determine propensity dependencies
        for (int r = 0; r < nr; ++r) {
            const auto &rxn = reactions[r];

            // Dependencies from reactant species (direct population dependency)
            std::set<int> deps;
            for (int ri : rxn.reactant_indices) {
                int si = ri - 1; // 1-based → 0-based
                if (si >= 0 && si < ns) {
                    deps.insert(si);
                }
            }

            // For MichaelisMenten: also depends on reactant species (already added above)

            // For Functional: additionally depends on all observable species
            if (rxn.rate_law_type == RateLawType::Functional) {
                has_functional_rates = true;
                for (int s = 0; s < ns; ++s) {
                    if (is_observable_species[s]) {
                        deps.insert(s);
                    }
                }
            }

            // GH #14 — under PSA the scaled propensity's leap factor (iScaling) is
            // bounded by the min population over reactants ∪ products, so a change
            // to any product population can change this reaction's effective
            // propensity. Register that dependency (PSA only).
            if (psa_product_deps) {
                for (int pi : rxn.product_indices) {
                    int si = pi - 1;
                    if (si >= 0 && si < ns) {
                        deps.insert(si);
                    }
                }
            }

            // Register this reaction's dependency on each species
            for (int s : deps) {
                species_to_reactions[s].push_back(r);
            }
        }

        // Remove duplicates in species_to_reactions (shouldn't have any with set, but be safe)
        for (auto &vec : species_to_reactions) {
            std::sort(vec.begin(), vec.end());
            vec.erase(std::unique(vec.begin(), vec.end()), vec.end());
        }

        // 3. For each reaction, determine which species change when it fires
        for (int r = 0; r < nr; ++r) {
            const auto &rxn = reactions[r];
            std::set<int> affected;
            for (int ri : rxn.reactant_indices) {
                int si = ri - 1;
                if (si >= 0 && si < ns && !species_list[si].fixed) {
                    affected.insert(si);
                }
            }
            for (int pi : rxn.product_indices) {
                int si = pi - 1;
                if (si >= 0 && si < ns && !species_list[si].fixed) {
                    affected.insert(si);
                }
            }
            reaction_to_affected_species[r].assign(affected.begin(), affected.end());
        }

        // 4. GH #190 — precompute the per-fired-reaction affected-reaction sets.
        //    For each reaction r, the reactions needing a propensity refresh after
        //    r fires are the union, over r's changed species, of species_to_reactions.
        //    Topology is fixed for the run, so derive each sorted/deduped set once
        //    here instead of re-deriving it (clear + insert + sort + unique) every
        //    step in the hot loop.
        reaction_to_affected_reactions.assign(nr, {});
        for (int r = 0; r < nr; ++r) {
            auto &out = reaction_to_affected_reactions[r];
            for (int s : reaction_to_affected_species[r]) {
                const auto &rxns = species_to_reactions[s];
                out.insert(out.end(), rxns.begin(), rxns.end());
            }
            std::sort(out.begin(), out.end());
            out.erase(std::unique(out.begin(), out.end()), out.end());
        }
    }

    // GH #190 — reactions needing a propensity recompute after `fired` executes:
    // a const reference into the table precomputed at build() (sorted, deduped),
    // so the SSA hot loop allocates and sorts nothing per step.
    const std::vector<int> &affected_reactions(int fired) const {
        return reaction_to_affected_reactions[fired];
    }
};

// ─── SsaSimulator::Impl ─────────────────────────────────────────────────────

// "R<index> (A + B -> C)": the label a reaction is reported under — the .net's
// 1-based index, then the reaction written in species names. Shared by the
// GH #110 reverse-fire diagnostic and the GH #616 reaction-statistics axis.
static std::string reaction_label(const NetworkModel &model, const Reaction &rxn) {
    const auto &names = model.species_names();
    auto join = [&](const std::vector<int> &idx1) {
        std::string s;
        for (int idx : idx1) {
            if (idx <= 0)
                continue; // a .net null species ("0" on the reaction line)
            if (!s.empty())
                s += " + ";
            int si = idx - 1;
            s += (si < static_cast<int>(names.size())) ? names[si] : "?";
        }
        return s.empty() ? std::string("0") : s;
    };
    return "R" + std::to_string(rxn.index) + " (" + join(rxn.reactant_indices) + " -> " +
           join(rxn.product_indices) + ")";
}

// Issue #809 — refuse a NaN or infinite propensity, naming the reaction (or,
// with r < 0, the total) and the time. Out of line and cold: it is reached at
// most once per run, and building the message inside the SSA loop's lambdas
// made them too large to inline, which cost 8-22% on small networks.
[[noreturn]] BNGSIM_SSA_COLD static void
throw_nonfinite_propensity(const NetworkModel &model, int r, double value, double t_at, bool psa) {
    // Spelled out for NaN and infinity: glibc prints a NaN whose sign bit is
    // set as "-nan" and macOS prints "nan", and the message should not depend
    // on the platform (or on the sign bit, which carries no meaning).
    auto num = [](double v) {
        if (std::isnan(v))
            return std::string("nan");
        if (std::isinf(v))
            return std::string(v > 0 ? "inf" : "-inf");
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.17g", v);
        return std::string(buf);
    };
    const std::string what = r >= 0 ? "the propensity of reaction " +
                                          reaction_label(model, model.reactions()[r]) + " is " +
                                          num(value)
                                    : "the total propensity is " + num(value) +
                                          " (every reaction's is finite; their sum overflowed)";
    throw std::runtime_error(
        std::string(psa ? "PSA: " : "SSA: ") + what + " at t = " + num(t_at) +
        ". A rate law that evaluates to NaN or infinity has no stochastic meaning: check its "
        "parameters and functions at this state (for example a sqrt or log of a negative value, "
        "a division by zero, or a table function read at a NaN index).");
}

struct SsaSimulator::Impl {
    NetworkModel &model;

    // GH #135 (fix 1a) — the dependency graph is a pure function of the model
    // topology (reactions, observable membership, fixed-species flags), which is
    // fixed for a simulator's lifetime: it binds one model reference at
    // construction. Building it (sets, sorts, the functional-rate all-observable-
    // species expansion) is O(nr·deps) and was previously redone on every run().
    // For an SSA ensemble that re-runs the same simulator across replicates this
    // dominated the per-replicate cost on low-activity models, where the step
    // loop barely runs. Cache it here, build lazily on first run, and reuse it.
    // The sum tree holds per-state propensities and is still built per-run
    // (its O(nr) allocation is trivial next to the graph build).
    // Issue #718 — which state slots hold molecule counts (1) rather than a
    // continuous quantity (0). Structural like the graph, so built once.
    std::vector<char> count_slot;

    DependencyGraph dep_graph;
    bool dep_graph_built = false;
    int dep_graph_ns = -1; // topology fingerprint guarding stale reuse
    int dep_graph_nr = -1;
    bool dep_graph_psa = false; // GH #14 — PSA adds product-population deps

    // GH #190 — path to the cc-compiled value-specialized propensity-vector .so
    // (symbol bngsim_ssa_propensities), produced by the Python codegen layer
    // (_codegen.prepare_ssa_propensity_lib) and content-cached on disk. When set
    // and the model is recompute-all eligible (pure mass-action exact SSA, no
    // events, small nr), run_internal loads it and takes the RR-style
    // recompute-all + flat-scan loop by DEFAULT — no MIR required. Empty ⇒ the
    // model wasn't eligible / codegen was skipped, and the incremental sum-tree
    // path runs unchanged.
    std::string propensity_lib_path;

    // Issue #719 — see SsaSimulator::set_breakpoints. Sorted, unique.
    std::vector<double> breakpoints;
    // Functions constant in time between the breakpoints (set_piecewise_constant_functions).
    std::vector<std::string> pc_functions;

    // GH #616 — record per-reaction firing counts and propensity integrals at
    // every output time (set_record_reaction_stats). The labels the result
    // reports the reaction axis under are built once per simulator, on the
    // first run that asks for them.
    bool record_reaction_stats = false;
    std::vector<std::string> reaction_labels;

    Impl(NetworkModel &m) : model(m) {}

    // Build the dependency graph if not already cached for the current topology.
    // The (ns, nr) fingerprint defends against a model whose structure changed
    // under the simulator (unusual — topology is normally fixed once loaded).
    DependencyGraph &dependency_graph(bool use_psa) {
        const int ns = model.n_species();
        const int nr = model.n_reactions();
        if (!dep_graph_built || dep_graph_ns != ns || dep_graph_nr != nr ||
            dep_graph_psa != use_psa) {
            dep_graph.build(model, use_psa);
            dep_graph_built = true;
            dep_graph_ns = ns;
            dep_graph_nr = nr;
            dep_graph_psa = use_psa;
        }
        return dep_graph;
    }
};

// The molecule count a stored value stands for (issue #692): storage is n/V, so
// n = c·V, which lands within rounding of a whole number when n is one. Such a
// count is put back on the whole number; a fractional one (an event can assign
// it) is returned as it is. `+ 0.0` turns a -0.0 into 0.
static double storage_to_count(double storage_value, double volume_factor) {
    const double n = storage_value * volume_factor;
    const double whole = std::round(n);
    if (std::fabs(n - whole) <=
        8.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::fabs(whole)))
        return whole + 0.0;
    return n;
}

static double round_initial_population_to_storage(double storage_value, double volume_factor) {
    if (!std::isfinite(storage_value) || !std::isfinite(volume_factor) || volume_factor <= 0.0) {
        return storage_value;
    }

    double amount = storage_value * volume_factor;
    double rounded_amount = (amount >= 0.0) ? std::floor(amount + 0.5) : std::ceil(amount - 0.5);
    return rounded_amount / volume_factor;
}

// ─── Public interface ────────────────────────────────────────────────────────

SsaSimulator::SsaSimulator(NetworkModel &model) : impl_(std::make_unique<Impl>(model)) {}

SsaSimulator::~SsaSimulator() = default;

void SsaSimulator::set_record_reaction_stats(bool enabled) {
    impl_->record_reaction_stats = enabled;
}

void SsaSimulator::set_piecewise_constant_functions(const std::vector<std::string> &names) {
    impl_->pc_functions = names;
}

void SsaSimulator::set_breakpoints(const std::vector<double> &times) {
    auto &bp = impl_->breakpoints;
    bp.clear();
    for (double x : times)
        if (std::isfinite(x))
            bp.push_back(x);
    std::sort(bp.begin(), bp.end());
    bp.erase(std::unique(bp.begin(), bp.end()), bp.end());
}

void SsaSimulator::set_propensity_library(const std::string &so_path) {
    impl_->propensity_lib_path = so_path;
}

Result SsaSimulator::run(const TimeSpec &times, uint64_t seed, double timeout_seconds) {
    // poplevel = 0.0 means no scaling (exact SSA)
    return run_internal(times, seed, 0.0, timeout_seconds);
}

Result SsaSimulator::run_psa(const TimeSpec &times, uint64_t seed, double poplevel,
                             double timeout_seconds) {
    if (poplevel <= 1.0) {
        throw std::invalid_argument(
            "PSA poplevel (N_c) must be > 1. Got " + std::to_string(poplevel) +
            ". For exact stochastic simulation, use run() instead of run_psa().");
    }
    return run_internal(times, seed, poplevel, timeout_seconds);
}

// ─── Unified SSA/PSA simulation loop ─────────────────────────────────────────
//
// When poplevel = 0: exact SSA (no scaling).
// When poplevel > 1: PSA with N_c = poplevel.
//
// PSA Algorithm 1 from Lin, Feng, Hlavacek (2019):
//   For each reaction r:
//     N_min^r = min population over reactants ∪ products (Eq. 14 + run_network's
//               default product-scale check; GH #14). Reactants bound depletion,
//               products bound overshoot of a small produced species. Synthesis
//               (∅ → A) has no reactants, so A alone bounds it.
//     λ_r = 1 / max(1, ⌊N_min^r / N_c⌋)
//     scaled_rate_r = λ_r * propensity_r
//   When reaction r fires:
//     Update each species s by (1/λ_r) * ξ_{r,s}
//
// Optimizations:
//   - Dependency graph: only recompute propensities for affected reactions
//   - Sum tree: O(log N) reaction selection + O(log N) propensity updates

Result SsaSimulator::run_internal(const TimeSpec &times, uint64_t seed, double poplevel,
                                  double timeout_seconds) {
    auto &model = impl_->model;
    WallClockBudget budget(timeout_seconds);
    const int ns = model.n_species();
    const int nr = model.n_reactions();
    const int n_obs = model.n_observables();
    const bool use_psa = (poplevel > 1.0);

    if (ns == 0) {
        throw std::runtime_error("Cannot simulate: model has no species");
    }

    // GH #106: rateOf(species) is the instantaneous ODE derivative dx/dt, which
    // has no well-defined meaning in a discrete stochastic trajectory (libRoad-
    // Runner likewise only evaluates rateOf for deterministic ODE simulation).
    // Reject loudly rather than silently feeding a stale/zero derivative into a
    // trigger or rate-rule term. Use method="ode" for rateOf models.
    if (model.uses_rateof()) {
        throw std::runtime_error(
            "This model uses the SBML rateOf csymbol (instantaneous dx/dt), which "
            "is only supported for ODE simulation (method=\"ode\"); rateOf has no "
            "well-defined value in a stochastic (SSA) trajectory.");
    }

    // Reject delay-bearing events: SSA and PSA run non-delayed events only, and
    // adding the pending-execution queue the ODE engine has is issue #526.
    // Surface a clear error here rather than at Simulator construction so the
    // message includes the event id. run() and run_psa() both come through here.
    // The message used to send the reader to "Phase 5c of the SBML SSA Support
    // Plan", a planning document that is no longer in the tree (issue #526).
    //
    // A delay expression that reads only fixed parameters and is 0 for this
    // run queues nothing, exactly as a literal 0 does. The SBML loader keeps
    // `<delay><ci>d</ci></delay>` as an expression so set_param reaches it
    // (issue #835), and before that it folded to a literal 0 that ran here.
    {
        const auto &evs = model.events();
        for (std::size_t ei = 0; ei < evs.size(); ++ei) {
            const auto &ev = evs[ei];
            const bool delayed = ev.delay_expr_idx >= 0
                                     ? !model.event_delay_is_fixed_zero(static_cast<int>(ei))
                                     : ev.delay > 0.0;
            if (delayed) {
                throw std::runtime_error("Event '" + ev.id +
                                         "' has a delay, which is not yet supported under SSA/PSA "
                                         "(issue #526 tracks adding it). Use method='ode', which "
                                         "runs delayed events, or remove the delay.");
            }
        }
    }

    // Per-instance RNG with deterministic seed
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    // Random tie-break among equal-priority simultaneous events (SBML L3v2
    // §4.11.6, issue #755). A stream of its own, derived from the run seed, so
    // it is reproducible per seed and a tie never shifts the reaction stream:
    // a model with no tie draws nothing from it and runs exactly as before.
    std::mt19937_64 event_rng(seed ^ 0x9E3779B97F4A7C15ULL);

    // GH #149 ablation — fast RNG (BNGSIM_SSA_FAST_RNG=1). std::mt19937_64 +
    // uniform_real_distribution is the slow std combo; SSA draws 2 uniforms +
    // a log per step, so the RNG is a classic per-step cost. xoshiro256++ seeded
    // via splitmix64 produces a [0,1) double in a handful of instructions. This
    // changes the random stream (a different—but valid—SSA realization), so it is
    // a TIMING ablation only: it isolates how much of the per-step cost is the
    // generator vs everything else. OFF by default (bit-identical std path).
    uint64_t xs[4];
    {
        uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
        for (int i = 0; i < 4; ++i) {
            z += 0x9E3779B97F4A7C15ULL;
            uint64_t w = z;
            w = (w ^ (w >> 30)) * 0xBF58476D1CE4E5B9ULL;
            w = (w ^ (w >> 27)) * 0x94D049BB133111EBULL;
            xs[i] = w ^ (w >> 31);
        }
    }
    const bool fast_rng = [] {
        const char *f = std::getenv("BNGSIM_SSA_FAST_RNG");
        return f && f[0] != '\0' && f[0] != '0';
    }();

    // GH #190 — reaction-selection structure (opt-in; default sum tree). bngsim
    // selects with a 4-ary sum tree (issue #713; it was a Fenwick tree): O(1)
    // total, O(log n) find, plus an O(log n) update
    // per affected reaction. For SMALL reaction counts a flat cumulative array
    // (O(n) total + O(n) linear scan, but a single contiguous, branch-predictable,
    // L1-resident pass — RoadRunner's direct method) wins the per-selection
    // constant factors: the microbench (dev/notes/gh190_select_microbench.cpp)
    // measured 1.4-1.9x on the isolated selection workload for nr<=44. BUT once
    // the per-step affected-set sort/dedup is precomputed (below), selection is a
    // small fraction of per-step cost and the flat win washes out end-to-end
    // (flat≈tree within noise on the high-activity suite models). So the flat
    // path stays OPT-IN — validated and bit-identical, useful for selection-bound
    // regimes and ablation, but not the default (no measured end-to-end gain, and
    // the index-order sum is a different—if equivalent—realization). Select with:
    //   unset/"fenwick"   sum tree (default; the name predates issue #713)
    //   "flat"            force the flat array
    //   "auto"            size-adaptive: flat when nr <= FLAT_SELECT_MAX_NR
    constexpr int FLAT_SELECT_MAX_NR = 64;
    const bool use_flat_select = [&] {
        const char *f = std::getenv("BNGSIM_SSA_SELECT");
        if (!f)
            return false;
        if (std::strcmp(f, "flat") == 0)
            return true;
        if (std::strcmp(f, "auto") == 0)
            return nr <= FLAT_SELECT_MAX_NR;
        return false; // "fenwick" / unrecognized → sum tree (default)
    }();
    auto next_u01 = [&]() -> double {
        if (!fast_rng)
            return uniform(rng);
        auto rotl = [](uint64_t x, int k) { return (x << k) | (x >> (64 - k)); };
        const uint64_t result = rotl(xs[0] + xs[3], 23) + xs[0];
        const uint64_t t = xs[1] << 17;
        xs[2] ^= xs[0];
        xs[3] ^= xs[1];
        xs[1] ^= xs[2];
        xs[0] ^= xs[3];
        xs[2] ^= t;
        xs[3] = rotl(xs[3], 45);
        return (result >> 11) * (1.0 / 9007199254740992.0); // 2^-53
    };

    // Output time points
    std::vector<double> t_out = times.output_times();
    const int n_out = static_cast<int>(t_out.size());

    // Working arrays: species populations (0-based). conc[] holds each species
    // in its storage units, n/V, which is what rate laws, observables and the
    // recorded trajectory read. counts[] holds the molecule count n itself, which
    // is what a firing changes (issue #692, see fire_species).
    //
    // Only a molecule count is rounded to a whole number (issue #718). The SBML
    // loader also gives a state slot to quantities that are not counts: a
    // parameter, compartment or stoichiometry symbol that an event or a rate
    // rule writes, and an assignment- or rate-rule target. It marks them
    // Species::continuous. Rounding every slot ran a rate constant of 0.5 as 1
    // and 0.4 as 0, a volume of 0.4 as 0, and put a rate-rule variable back on
    // a whole number at the start of every run_until leg. A rate-rule target
    // built without the mark is recognised by its reaction. A fixed (boundary)
    // species is a count and is rounded, as run_network rounds every species
    // before an SSA. Each count this moves is reported
    // (ssa_diagnostics.n_rounded_populations), on every leg: the Python
    // validator only sees the state at Simulator construction.
    if (static_cast<int>(impl_->count_slot.size()) != ns) {
        auto &is_count = impl_->count_slot;
        is_count.assign(ns, 1);
        for (int i = 0; i < ns; ++i)
            if (model.species()[i].continuous)
                is_count[i] = 0;
        for (const auto &rxn : model.reactions())
            if (rxn.is_rate_rule_ode)
                for (int pi : rxn.product_indices)
                    if (pi >= 1 && pi <= ns)
                        is_count[pi - 1] = 0;
    }
    const std::vector<char> &is_count = impl_->count_slot;
    long n_rounded = 0;
    int first_rounded = -1;
    std::vector<double> conc(ns);
    std::vector<double> counts(ns);
    // A run that continues the previous one (issue #693) starts from that run's
    // own state, which may hold a fractional count an event assigned (kept, as
    // store_value keeps it): rounding it here made a run split into legs differ
    // from the run whole. Only a fresh start's populations are rounded.
    const bool continuing = model.event_carry_for(times.t_start, model.n_events()) != nullptr;
    for (int i = 0; i < ns; ++i) {
        const auto &sp = model.species()[i];
        if (!is_count[i] || continuing) {
            conc[i] = sp.concentration;
            counts[i] = conc[i] * sp.volume_factor; // a continuous value, not snapped
            continue;
        }
        conc[i] = round_initial_population_to_storage(sp.concentration, sp.volume_factor);
        counts[i] = storage_to_count(conc[i], sp.volume_factor);
        // The same test _ssa_validation applies: a count that moved by more
        // than conversion noise, relative to the count, was fractional. It is
        // relative all the way down, so 5e-10 of a molecule rounded to 0 (a
        // model written in moles) is reported rather than read as noise.
        const double amount = sp.concentration * sp.volume_factor;
        const double whole = conc[i] * sp.volume_factor;
        if (std::isfinite(amount) && std::fabs(amount - whole) > 1e-9 * std::fabs(amount)) {
            ++n_rounded;
            if (first_rounded < 0)
                first_rounded = i;
        }
    }

    // Propensity arrays
    std::vector<double> propensities(nr, 0.0);    // unscaled propensities (magnitudes)
    std::vector<double> scaling_factors(nr, 1.0); // λ_r per reaction (PSA)

    // GH #15 — PSA partial-scaling diagnostics (accumulators; only used when
    // use_psa). The (m_r, a_r) pair in force for each reaction is banked lazily:
    // whenever set_propensity recomputes reaction r it first flushes the previous
    // pair over [psa_last_t[r], psa_now] into the dwell integrals, then records
    // the new pair. A closing sweep after the loop flushes the tail. This is
    // O(affected) per step — exactly the reactions the dep-graph already
    // recomputes — plus one O(nr) sweep at the end. psa_now mirrors the current
    // simulated time; it is updated whenever `t` advances so every set_propensity
    // call flushes against the correct clock.
    const std::size_t psa_n = use_psa ? static_cast<std::size_t>(nr) : 0;
    std::vector<double> psa_m_at(psa_n, 1.0);             // m_r currently in force
    std::vector<double> psa_a_at(psa_n, 0.0);             // unscaled a_r in force
    std::vector<double> psa_last_t(psa_n, times.t_start); // last flush time per rxn
    std::vector<double> psa_mbar_int(psa_n, 0.0);         // ∫ m_r dt
    std::vector<double> psa_qexc_int(psa_n, 0.0);         // ∫ (m_r−1)·a_r dt
    double psa_exact_int = 0.0;                           // ∫ Σ_r a_r dt
    double psa_scaled_int = 0.0;                          // ∫ Σ_r a_r/m_r dt
    double psa_peak_pop = 0.0;                            // running max participating count
    double psa_now = times.t_start;                       // mirror of current sim time
    auto psa_flush = [&](int r) {
        const double dt = psa_now - psa_last_t[r];
        if (dt > 0.0) {
            const double m = psa_m_at[r];
            const double a = psa_a_at[r];
            psa_mbar_int[r] += m * dt;
            psa_qexc_int[r] += (m - 1.0) * a * dt;
            psa_exact_int += a * dt;
            psa_scaled_int += (a / m) * dt; // m >= 1 by construction
        }
        psa_last_t[r] = psa_now;
    };

    // GH #110 — sign-split firing direction per reaction. +1 forward (rate >= 0,
    // reactants → products), -1 reverse (rate < 0, products → reactants). The
    // sum tree and propensities[] hold |rate| for selection; rxn_dir[r]
    // records which way reaction r runs at its last propensity evaluation.
    std::vector<int> rxn_dir(nr, 1);

    // GH #110 — boundary diagnostics accumulated over the run, surfaced as one
    // filterable warning per case by the Python layer. Indices resolved to
    // names after the loop (out of the hot path).
    long neg_cross_count = 0;    // species count crossings from >= 0 to < 0
    int first_neg_species = -1;  // 0-based species index of the first crossing
    long reverse_fire_count = 0; // reactions fired in reverse (negative rate)
    int first_reverse_rxn = -1;  // 0-based reaction index of the first reverse fire

    const int n_func = model.n_functions();
    const auto &reactions = model.reactions();
    const auto &species_list = model.species();
    const auto &events = model.events();
    const int n_events = static_cast<int>(events.size());

    // Apply a firing's change of `dn` molecules to species si (issue #692).
    //
    // A species is stored as n/V (V = its volume_factor), so a firing moves the
    // stored value by dn/V. It used to add dn/V to the stored value, and every
    // such add rounded, and the rounding walked: after 1e5 firings at V = 10 an
    // extinct species sat at -1.9e-7 molecules, which the negative-count
    // diagnostic reported as a crossing, and a residue of the other sign left it
    // a propensity it could fire on. The firing now moves the count, which is a
    // whole number held exactly, and the stored value is derived from it with
    // the one division the old update also made. So a count n is always stored
    // as the same double n/V, the one round_initial_population_to_storage and
    // the falling factorial's j/V (model.cpp) form, which makes the j-th factor
    // exactly 0 at n = j. At V = 1 (every `.net` model) the stored value is the
    // count, and the arithmetic is the old add, bit for bit.
    auto fire_species = [&](int si, double dn) {
        const double before = conc[si];
        counts[si] += dn;
        conc[si] = counts[si] / species_list[si].volume_factor;
        if (before >= 0.0 && conc[si] < 0.0) {
            ++neg_cross_count;
            if (first_neg_species < 0)
                first_neg_species = si;
        }
    };
    // An event assignment writes a stored value through here, so the count
    // stays the count of the stored value. A value within rounding of a whole
    // count is stored as exactly n/V; a fractional one is kept as written.
    auto store_value = [&](int si, double value) {
        const double n = storage_to_count(value, species_list[si].volume_factor);
        counts[si] = n;
        conc[si] = n == std::round(n) ? n / species_list[si].volume_factor : value;
    };

    // ─── Build (or reuse) dependency graph ───────────────────────────────────
    // Cached on the Impl across run() calls — topology-only, seed-independent.
    DependencyGraph &dep_graph = impl_->dependency_graph(use_psa);

    // ─── Build the selection tree ────────────────────────────────────────────
    // Stores effective propensities: unscaled for SSA, scaled for PSA.
    SumTree ftree(nr);

    // GH #190 — flat cumulative-propensity array (mirrors what goes into the
    // sum tree). Allocated only on the flat selection path; sel_set/sel_total/
    // sel_find below dispatch to one structure or the other.
    std::vector<double> flat_eff;
    if (use_flat_select)
        flat_eff.assign(nr, 0.0);
    auto sel_set = [&](int r, double value) {
        if (use_flat_select)
            flat_eff[r] = value;
        else
            ftree.set(r, value);
    };
    auto sel_total = [&]() -> double {
        if (!use_flat_select)
            return ftree.total();
        double s = 0.0; // fresh O(n) sum each step (drift-free, RR's direct method)
        for (int r = 0; r < nr; ++r)
            s += flat_eff[r];
        return s;
    };
    // The index whose slice [running sum before it, running sum through it)
    // holds `target`. The flat path scans left-to-right in index order (the
    // same selection semantics as the tree's descent), terminating early once
    // the running sum passes the target; for small nr this contiguous pass
    // beats the tree's descent. The comparison is strict so a weight-0 entry,
    // whose slice is empty, is never taken (issue #713).
    auto sel_find = [&](double target) -> int {
        if (!use_flat_select)
            return ftree.find(target);
        double cum = 0.0;
        int last_positive = nr - 1;
        for (int r = 0; r < nr; ++r) {
            cum += flat_eff[r];
            if (flat_eff[r] > 0.0)
                last_positive = r;
            if (cum > target)
                return r;
        }
        return last_positive;
    };

    // Issue #809 — a propensity that is NaN or infinite has no stochastic
    // meaning, and the loop cannot recover from one: a NaN total fails the
    // `a0 <= 0` stuck test and gives a NaN waiting time, so the SSA never
    // reached another sample and spun forever, while PSA returned the frozen
    // initial state as a trajectory. Refuse it where it is computed, naming the
    // reaction and the time, as the ODE path refuses a non-finite RHS.
    auto refuse_nonfinite_propensity = [&](int r, double value, double t_at) {
        throw_nonfinite_propensity(model, r, value, t_at, use_psa);
    };

    // ─── Structure-specialized propensity-vector backend (GH #149 / #190) ─────
    // RoadRunner fills the whole propensity vector with one native call; bngsim
    // evaluates per-reaction (compute_propensity), which #149 profiled as the
    // dominant per-step cost. The structure-specialized vector
    // (emit_ssa_propensity_source_structure: rate constants read from a runtime
    // params[] arg, only the structural stat·svf factor baked) is compiled ONCE
    // per model and reused across every parameter point (a fit) and replicate (an
    // ensemble) — no per-point recompile. It can be supplied three ways, all
    // resolving to one prop_jit_fn pointer so the loops stay backend-agnostic:
    //   • DEFAULT (no env): a cc-compiled .so handed down by the Python codegen
    //     layer (set_propensity_library) — engages the RR-style recompute-all
    //     loop for eligible small mass-action exact models. No MIR needed.
    //     Compiler-less fallback: if no cc .so was supplied (the host has no C
    //     compiler) but this build embeds MIR, the same vector is JIT'd
    //     in-process via MirJit so compiler-less + MIR-built distributions still
    //     get the fast path (GH #139/#140 role) instead of interpreted.
    //   • BNGSIM_SSA_PROP_CC=1: compile in-process via cc (CcJit). Ablation.
    //   • BNGSIM_SSA_PROP_JIT=1: in-process MIR JIT (MirJit, needs a MIR build).
    // The actual decision is made below, AFTER the event / rate-rule / time-
    // dependent gates are known; the declarations live here so the propensity
    // lambdas can capture them. Any setup failure falls back to compute_propensity.
    bool prop_jit_active = false;
    bool ssa_fast_loop = false; // RR-style recompute-all + flat-scan branch
    // GH #190 — the propensity backend actually used this run, recorded into the
    // result's SsaDiagnostics for accurate reporting. "cc"/"mir" once a compiled
    // kernel is in use, else "interpreted".
    std::string prop_backend = "interpreted";
    std::vector<double> a_jit;
    // GH #190 — the structure-specialized kernel reads each reaction's rate
    // constant from this runtime parameter-value buffer rather than from a baked
    // literal, so ONE compiled kernel serves every parameter point (a fit) and
    // every replicate (an ensemble) with no per-point recompile. Snapshotted from
    // the model at setup and refreshed after any event that can mutate a parameter.
    std::vector<double> param_vals;
    void (*prop_jit_fn)(const double *, const double *, double *) = nullptr;
    std::unique_ptr<MirJit> prop_jit;
    std::unique_ptr<CcJit> prop_cc;
    std::unique_ptr<DynamicLibrary> prop_lib;
    auto refresh_param_vals = [&]() {
        const auto &ps = model.parameters();
        param_vals.resize(ps.size());
        for (std::size_t i = 0; i < ps.size(); ++i)
            param_vals[i] = ps[i].value;
    };
    // Refill the whole propensity buffer from current conc[] (one native call).
    auto refresh_jit_propensities = [&]() {
        if (prop_jit_active)
            prop_jit_fn(conc.data(), param_vals.data(), a_jit.data());
    };

    // Allocate result
    Result result;
    result.allocate(n_out, ns, n_obs);
    result.set_species_names(model.species_names());
    // GH #71: project trajectory columns to reported species only when the
    // model has unreported state (event-mutated parameter/compartment promoted
    // to a species). All-reported models leave the projection empty, byte-
    // identical column set.
    {
        auto reported = model.reported_species_indices();
        if (reported.size() != static_cast<std::size_t>(model.n_species())) {
            result.set_reported_species_indices(std::move(reported));
        }
    }
    result.set_observable_names(model.observable_names());
    if (n_func > 0) {
        result.set_expression_names(model.function_names());
    }

    // Observable buffer
    std::vector<double> obs_buf(n_obs);

    // ─── GH #616: per-reaction firing counts and propensity integrals ─────────
    //
    // Opt-in (set_record_reaction_stats). For each reaction r the run keeps
    //   rs_count[r]     — fires of r so far (a reverse fire is a fire of r),
    //   rs_a[r]         — |a_r| in force since rs_last_t[r],
    //   rs_integral[r]  — ∫ |a_r| ds from t_start to rs_last_t[r],
    // so that ∫_{t_start}^{t} |a_r| ds = rs_integral[r] + rs_a[r]·(t − rs_last_t[r])
    // at any t ≥ rs_last_t[r] before a_r next changes. The integral is banked
    // lazily, at the moment a propensity is REWRITTEN (set_propensity, or the
    // fast loop's per-step refill), so the incremental dependency-graph path
    // pays O(affected) per step for this exactly as it does for the propensities
    // themselves; the read-out at an output time is one O(nr) pass. |a_r| is the
    // magnitude the selection used (GH #110 sign-split), so N_r − ∫|a_r| is the
    // compensated counting process of the channel AS SAMPLED, and its mean is
    // zero — the identity the Python tests hold the instrumentation to. PSA is
    // excluded: a scaled channel fires m_r molecules at intensity a_r/m_r, which
    // is not the exact process these statistics describe.
    const bool rec_stats = impl_->record_reaction_stats && !use_psa;
    std::vector<double> rs_count, rs_a, rs_last_t, rs_integral, rs_scratch;
    if (rec_stats) {
        rs_count.assign(nr, 0.0);
        rs_a.assign(nr, 0.0);
        rs_last_t.assign(nr, times.t_start);
        rs_integral.assign(nr, 0.0);
        rs_scratch.assign(nr, 0.0);
        result.allocate_reaction_stats(n_out, nr);
        if (impl_->reaction_labels.size() != static_cast<std::size_t>(nr)) {
            impl_->reaction_labels.clear();
            impl_->reaction_labels.reserve(nr);
            for (int r = 0; r < nr; ++r)
                impl_->reaction_labels.push_back(reaction_label(model, reactions[r]));
        }
        result.set_reaction_labels(impl_->reaction_labels);
    }
    // Bank the propensity in force over the dwell ending at t_now, then install
    // the one taking effect.
    auto rs_bank = [&](int r, double t_now, double a_new) {
        rs_integral[r] += rs_a[r] * (t_now - rs_last_t[r]);
        rs_last_t[r] = t_now;
        rs_a[r] = a_new;
    };
    // Write the statistics as of output time t_at into result row idx.
    auto rs_record = [&](int idx, double t_at) {
        for (int r = 0; r < nr; ++r)
            rs_scratch[r] = rs_integral[r] + rs_a[r] * (t_at - rs_last_t[r]);
        result.record_reaction_stats(idx, rs_count.data(), rs_scratch.data());
    };

    // ─── Event support (SBML L3) ─────────────────────────────────────────────
    //
    // Events under SSA mirror the cvode path's semantics (cvode_simulator.cpp
    // process_firing_batch + t=0 init), but trigger detection within a τ-step
    // is by bisection over (t, t+τ] rather than continuous root-find — state
    // is piecewise-constant during τ, so any time-dependent trigger is a
    // 1-D function of t and is well-resolved by bisection. State-dependent
    // triggers can only flip at fire boundaries (state is constant during τ);
    // a post-fire trigger sweep handles those.
    //
    // Delayed events never reach this loop: the check at the top of
    // run_internal() raises on any event whose delay is not a fixed zero
    // (NetworkModel::event_delay_is_fixed_zero; delay support under SSA/PSA is
    // issue #526).
    std::vector<bool> trigger_was_true(n_events, false);
    auto &eval_ref = model.evaluator();
    auto &sp_vec_ref = const_cast<std::vector<Species> &>(model.species());

    // Helper: write conc[] back to the species list and refresh observables /
    // functions at time `t_eval` so trigger and assignment-RHS expressions
    // see consistent state.
    auto sync_state = [&](double t_eval) {
        for (int si = 0; si < ns; ++si) {
            sp_vec_ref[si].concentration = conc[si];
        }
        model.update_observables(conc.data());
        model.evaluate_functions(t_eval);
    };

    // Issues #719/#751/#753 — the continuous path (see the loop for models with
    // time-dependent rates or rate rules, below). A *dynamic* reaction is one
    // whose propensity changes between firings: a Functional rate law, or a
    // reactant that is a rate-rule target. Its propensity is not held in the
    // selection tree (which then holds only the constant part of a0); the loop
    // integrates it over time instead. Empty, and cont_mode false, for every
    // model without time-dependent rates or rate rules.
    bool cont_mode = false;
    std::vector<char> is_dyn(nr, 0);
    std::vector<int> dyn;

    // GH #14 — the PSA leap m_r for reaction r at the current populations,
    // updating the peak-population diagnostic (GH #15).
    auto psa_leap = [&](int r) -> double {
        const auto &rxn = reactions[r];
        // GH #14 — PSA leap factor (iScaling = 1/λ_r) is governed by the
        // smallest population among ALL species the reaction changes: its
        // reactants (a leap cannot consume more than are present) AND its
        // products (a leap must not overshoot a currently-small produced
        // species). This is the union min over reactants ∪ products, matching
        // run_network's default heterogeneous adaptive scaling (rxn_rate_scaled
        // with pScaleChecker=true, Network3/network.cpp). A synthesis reaction
        // (∅ → A) has no reactants, so its bound comes from the product A: it
        // scales once A is large and runs as exact SSA while A is small —
        // unlike run_network, which scales synthesis by a fixed N_c regardless
        // of A. Direction is irrelevant here since both sides are inspected.
        double n_min = std::numeric_limits<double>::max();
        auto fold_min = [&](const std::vector<int> &idx) {
            for (int ci : idx) {
                int si = ci - 1;
                if (si >= 0 && si < ns) {
                    const double count = conc[si] * species_list[si].volume_factor;
                    n_min = std::min(n_min, count);
                    psa_peak_pop = std::max(psa_peak_pop, count); // GH #15
                }
            }
        };
        fold_min(rxn.reactant_indices);
        fold_min(rxn.product_indices);
        // Only a null reaction (no reactants and no products) leaves the
        // sentinel; it changes nothing, so leave it unscaled.
        if (n_min == std::numeric_limits<double>::max())
            n_min = 0.0;
        return std::max(1.0, std::floor(n_min / poplevel));
    };

    // Helper: (re)compute one reaction's propensity, direction, PSA scaling,
    // and sum-tree entry from the current conc[]. GH #110: the rate law is
    // evaluated literally and may be negative; we store |rate| for selection and
    // record the sign in rxn_dir[r] (+1 forward, -1 reverse). The selection
    // magnitude is direction-agnostic, so a reaction whose rate law goes
    // negative still contributes |rate| to a0 and, when selected, fires in
    // reverse (see the firing step) — making the SSA's expected drift equal the
    // ODE RHS at every state. The PSA leap bound (n_min) is taken over the
    // species being *consumed* in the active direction, which flips to the
    // products under reverse firing.
    auto set_propensity = [&](int r) {
        // GH #81 — a rate-rule ODE reaction (`dX/dt = f`, compiled to `[] → [X]`)
        // is NOT a stochastic channel: its target is integrated deterministically
        // (forward Euler) below. Keep it out of the tree selection so it never
        // contributes to a0 and is never picked as a fire.
        if (reactions[r].is_rate_rule_ode) {
            sel_set(r, 0.0);
            return;
        }
        // GH #81 (Tier 2) — an ODE-only reaction (the #86 concentration-dilution
        // term for a rate-rule compartment) is excluded from SSA entirely: it is
        // neither a stochastic channel nor integrated. Under SSA the molecule
        // count is conserved across a volume change; the volume's effect on
        // reaction rates is carried by the ssa_live_volume_* propensity
        // correction, not by diluting the stored count.
        if (reactions[r].ode_only) {
            sel_set(r, 0.0);
            return;
        }
        double signed_prop = prop_jit_active ? a_jit[r] : model.compute_propensity(r, conc.data());
        if (!std::isfinite(signed_prop))
            refuse_nonfinite_propensity(r, signed_prop, psa_now);
        int dir = (signed_prop < 0.0) ? -1 : 1;
        double prop = (dir < 0) ? -signed_prop : signed_prop; // |signed_prop|
        rxn_dir[r] = dir;
        propensities[r] = prop;

        if (cont_mode && is_dyn[r]) {
            // A dynamic reaction (see above): its propensity is integrated by the
            // continuous loop, not held in the tree. What is set here is what
            // stays fixed until the next firing: the PSA leap, from the
            // populations. The dwell integrals bank a zero rate; the loop adds
            // the reaction's integrated propensity itself.
            if (rec_stats)
                rs_bank(r, psa_now, 0.0);
            if (use_psa) {
                const double m_r = psa_leap(r);
                scaling_factors[r] = 1.0 / m_r;
                psa_flush(r);
                psa_m_at[r] = m_r;
                psa_a_at[r] = 0.0;
            }
            sel_set(r, 0.0);
            return;
        }
        if (rec_stats) // GH #616 — psa_now mirrors t (kept current on every advance)
            rs_bank(r, psa_now, prop);

        double effective_prop = prop;
        if (use_psa && prop > 0.0) {
            const double inv_lambda = psa_leap(r);
            scaling_factors[r] = 1.0 / inv_lambda;
            effective_prop = scaling_factors[r] * prop;
            // GH #15 — bank the (m_r, a_r) that was in force over its dwell, then
            // record the pair now taking effect. inv_lambda is exactly m_r.
            psa_flush(r);
            psa_m_at[r] = inv_lambda;
            psa_a_at[r] = prop;
        } else if (use_psa) {
            scaling_factors[r] = 1.0;
            effective_prop = 0.0;
            // GH #15 — reaction inactive (prop == 0): close its dwell, record the
            // idle pair (m=1, a=0) so the integrals see no contribution.
            psa_flush(r);
            psa_m_at[r] = 1.0;
            psa_a_at[r] = 0.0;
        }
        sel_set(r, effective_prop);
    };

    // Helper: recompute ALL propensities + sum-tree entries. Called after
    // an event fires (assignments can touch any species, so the dep-graph
    // narrow-update is unsafe).
    auto recompute_all_propensities = [&]() {
        if (prop_jit_active)
            refresh_param_vals();   // an event may have mutated a rate parameter
        refresh_jit_propensities(); // GH #149: refill a_jit[] from current conc[]
        for (int r = 0; r < nr; ++r)
            set_propensity(r);
    };

    // ─── GH #81: rate-rule ODE targets (deterministic sub-step integration) ──
    //
    // A reaction flagged is_rate_rule_ode was compiled from an SBML rate rule
    // `dX/dt = f` into the Functional reaction `[] → [X]`. Under ODE the engine
    // accumulates `derivs[X] += f` (÷V_c(X) for the per-species/hOSU case);
    // under exact SSA, X is a deterministic continuous quantity and must be
    // integrated the same way, NOT sampled as a stochastic birth/death channel
    // (that would inject Poisson noise, round X to integers, and — when |f| is
    // large — swamp the propensity sum). We collect the target of each such
    // reaction and advance it by forward Euler at each sub-step; the propensity
    // refresh that already runs for time-dependent rates then lets every
    // reaction reading X follow its continuous trajectory.
    struct RateRuleOde {
        int rxn;          // reaction index (for compute_propensity → RHS f)
        int target0;      // 0-based target species index (X)
        bool per_species; // hOSU target: divide f by V_c(X), mirroring compute_derivs
        double vf;        // V_c(X) = species volume_factor
    };
    std::vector<RateRuleOde> rate_rule_odes;
    for (int r = 0; r < nr; ++r) {
        const auto &rxn = reactions[r];
        if (!rxn.is_rate_rule_ode)
            continue;
        if (rxn.product_indices.empty())
            continue; // defensive: a rate rule always has exactly one product
        int t0 = rxn.product_indices[0] - 1;
        if (t0 < 0 || t0 >= ns)
            continue;
        rate_rule_odes.push_back(
            {r, t0, rxn.per_species_volume_scaling, species_list[t0].volume_factor});
    }
    const bool has_rate_rules = !rate_rule_odes.empty();

    // process_firing_batch — SBML L3v2 §4.11.6 simultaneous-event execution,
    // the same drain as cvode_simulator.cpp's process_firing_batch (GH #242).
    //
    // The batch is a dynamic MULTISET of execution instances, one per rising
    // edge, not the fixed list the caller hands in. This used to be a copy of
    // the ODE drain from before GH #242, and it differed in two ways, each
    // silent:
    //   - An assignment that turned another event's trigger true did not add it
    //     to the batch, and every caller then synced trigger_was_true to the
    //     post-batch truth, which recorded the new rising edge as "already
    //     true". A cascaded event never fired, at that instant or later
    //     (issue #761; SBML suite 00978 gave x, y, z = 0, 0, 0 against 5, 1, 3).
    //   - Among instances at the same maximum priority the lowest index always
    //     ran first, so the last-declared writer won every replicate and an
    //     ensemble collapsed onto one branch where §4.11.6 and the ODE engine
    //     pick at random (issue #755).
    //
    // So, as in the ODE drain:
    //  1. Seed one instance per event in `firing_in`; `prev` is the trigger
    //     baseline, with each seed marked true (it has just risen). Each
    //     instance freezes its useValuesFromTriggerTime values at its own
    //     trigger time: the pre-batch state for a seed.
    //  2. Drain highest priority first, re-evaluating priorities before every
    //     pick. Among not-done instances at the same maximum priority pick one
    //     at random from `event_rng`. A single candidate draws nothing, so a
    //     model with no tie never advances that stream.
    //  3. After each fire, refresh observables and functions and re-check every
    //     trigger against `prev`: a rising edge enqueues a new instance (its
    //     values frozen now), a falling edge cancels the not-done instances of
    //     a non-persistent event.
    //  4. CASCADE_LIMIT stops an algebraic loop (A arms B arms A ...).
    //  5. On exit trigger_was_true holds the settled truth of every trigger.
    // Delays never reach here: run_internal refuses a delayed event up front
    // (issue #526). Returns true if any event modified conc[].
    struct ExecInstance {
        int event_idx;
        std::vector<double> snapshot_vals; // UVFTT frozen RHS (empty if !UVFTT)
        bool done = false;
    };
    // Far above any legitimate cascade depth (00978 fires ~11; 01533 ~106).
    constexpr int CASCADE_LIMIT = 100000;
    auto process_firing_batch = [&](double t_now, const std::vector<int> &firing_in) -> bool {
        if (firing_in.empty())
            return false;

        auto make_instance = [&](int ei) -> ExecInstance {
            ExecInstance inst;
            inst.event_idx = ei;
            const auto &ev = events[ei];
            if (ev.use_values_from_trigger_time) {
                inst.snapshot_vals.reserve(ev.assignments.size());
                for (const auto &[sp_idx0, val_expr_idx] : ev.assignments) {
                    (void)sp_idx0;
                    inst.snapshot_vals.push_back(eval_ref.evaluate(val_expr_idx));
                }
            }
            return inst;
        };

        std::vector<bool> prev = trigger_was_true;
        std::vector<ExecInstance> queue;
        queue.reserve(firing_in.size());
        for (int ei : firing_in) {
            prev[ei] = true;
            queue.push_back(make_instance(ei));
        }

        auto eval_pri = [&](const ExecInstance &inst) -> double {
            const auto &ev = events[inst.event_idx];
            const double p = (ev.priority_expr_idx >= 0) ? eval_ref.evaluate(ev.priority_expr_idx)
                                                         : static_cast<double>(ev.priority);
            // NaN compares false with everything, so the order silently fell
            // back to declaration order. Refuse it.
            if (std::isnan(p))
                throw std::runtime_error("event '" + ev.id + "' has a priority that is NaN at t=" +
                                         std::to_string(t_now) +
                                         "; an event priority must be a number");
            return p;
        };

        bool any_fired = false;
        int fires = 0;
        std::vector<size_t> ties;
        while (true) {
            // Drop executed and cancelled instances once they pile up, so a
            // long cascade (an algebraic loop runs to CASCADE_LIMIT) costs
            // O(fires) rather than O(fires²). Order is kept, so the tie list
            // below is still in index order.
            if (queue.size() > 256)
                queue.erase(std::remove_if(queue.begin(), queue.end(),
                                           [](const ExecInstance &x) { return x.done; }),
                            queue.end());
            // The not-done instances sharing the maximum priority, in index
            // order, so a single candidate is the old lowest-index pick.
            double best_pri = 0.0;
            ties.clear();
            for (size_t k = 0; k < queue.size(); ++k) {
                if (queue[k].done)
                    continue;
                const double pk = eval_pri(queue[k]);
                if (ties.empty() || pk > best_pri) {
                    best_pri = pk;
                    ties.clear();
                    ties.push_back(k);
                } else if (pk == best_pri) {
                    ties.push_back(k);
                }
            }
            if (ties.empty())
                break;
            size_t k = ties[0];
            if (ties.size() > 1) {
                std::uniform_int_distribution<size_t> pick(0, ties.size() - 1);
                k = ties[pick(event_rng)];
            }
            queue[k].done = true;

            if (++fires > CASCADE_LIMIT) {
                throw std::runtime_error(
                    "Event cascade exceeded CASCADE_LIMIT at t=" + std::to_string(t_now) +
                    " (same-instant events appear to arm each other in an algebraic loop).");
            }

            const auto &ev = events[queue[k].event_idx];
            const auto &assigns = ev.assignments;
            std::vector<double> nv(assigns.size());
            if (ev.use_values_from_trigger_time) {
                nv = queue[k].snapshot_vals;
            } else {
                for (size_t a = 0; a < assigns.size(); ++a) {
                    nv[a] = eval_ref.evaluate(assigns[a].second);
                }
            }
            // A concentration assigned to a species stored as amount/V_static
            // in a moving compartment is stored as c·V_live/V_static, V_live as
            // it stands when this event executes, before its own assignments
            // (a resize among them included) apply (issue #741).
            for (size_t a = 0; a < assigns.size(); ++a) {
                const int sp = assigns[a].first;
                if (sp < 0 || sp >= ns)
                    continue;
                const Species &sv = model.species()[static_cast<size_t>(sp)];
                if (sv.ssa_live_volume_idx0 >= 0 && sv.ssa_live_volume_idx0 < ns)
                    nv[a] *= conc[sv.ssa_live_volume_idx0] / sv.volume_factor;
            }
            for (size_t a = 0; a < assigns.size(); ++a) {
                // GH #81 (Tier 1): skip ODE-only assignments under SSA. The
                // SBML loader marks the per-species `s := s·V_old/V_new`
                // concentration rescale injected at a compartment resize as
                // ode_only — under SSA the stored value is `amount/V_static`
                // and the count `conc·V_static` must stay unchanged, so the
                // rescale must not run (it would corrupt molecule counts). The
                // compartment's own resize assignment is NOT ode_only, so the
                // live volume the propensity correction reads still updates.
                if (a < ev.assignment_ode_only.size() && ev.assignment_ode_only[a])
                    continue;
                int sp_idx0 = assigns[a].first;
                if (sp_idx0 >= 0 && sp_idx0 < ns) {
                    // A NaN or infinite value has no stochastic meaning. One
                    // that a propensity reads is refused there (issue #809);
                    // one that nothing reads was carried into the trajectory,
                    // where the ODE path refuses it.
                    // "nan"/"inf" spelled out: std::to_string prints a NaN
                    // with its sign bit set as "-nan" on glibc.
                    if (!std::isfinite(nv[a]))
                        throw std::runtime_error(
                            std::string(use_psa ? "PSA" : "SSA") + ": event '" + ev.id +
                            "' assigns " +
                            (std::isnan(nv[a]) ? "nan" : (nv[a] > 0 ? "inf" : "-inf")) + " to " +
                            model.species()[sp_idx0].name + " at t=" + std::to_string(t_now) +
                            "; an event assignment must be a finite number");
                    store_value(sp_idx0, nv[a]);
                    sp_vec_ref[sp_idx0].concentration = conc[sp_idx0];
                }
            }
            any_fired = true;

            model.update_observables(conc.data());
            model.evaluate_functions(t_now);

            for (int ei = 0; ei < n_events; ++ei) {
                const bool now_true = eval_ref.evaluate(events[ei].trigger_expr_idx) > 0.5;
                if (now_true && !prev[ei]) {
                    queue.push_back(make_instance(ei));
                } else if (!now_true && prev[ei] && !events[ei].persistent) {
                    for (auto &inst : queue)
                        if (!inst.done && inst.event_idx == ei)
                            inst.done = true;
                }
                prev[ei] = now_true;
            }
        }

        trigger_was_true = prev;
        return any_fired;
    };

    // Helper: probe a trigger expression at a given time. Bisects within
    // (lo, hi] to find the rising-edge t_cross to bisect_tol precision. Pre:
    // trigger is FALSE at lo and TRUE at hi (with current conc[]). Returns
    // hi (the smallest known-true point).
    constexpr double BISECT_EPS = 1e-12;
    // Sample-vs-event-time tolerance. Several orders of magnitude wider
    // than BISECT_EPS so the recording loop defers a sample whose time is
    // within bisection imprecision of an event crossing, but not legitimate
    // samples strictly before the event. Defined at function scope so both
    // the a0==0 idle path and the τ-step recording loop can reuse it.
    constexpr double SAMPLE_EVENT_TOL = 1e-9;
    // Both tolerances are absolute, and at large t an absolute 1e-12 is below
    // the spacing of doubles: from t = 8192 one ulp is 2^-39 ≈ 1.8e-12, so
    // `hi - lo` can never fall under BISECT_EPS and the bisection below spun
    // forever at 100% CPU, never reaching the wall-clock budget, which is only
    // checked in the outer loops (issue #716). Floor each at one ulp of the
    // time it is applied at. Below t = 8192 one ulp is under 1e-12, so the
    // floor never engages there and every tolerance is exactly what it was
    // (two ulps would not be: from t = 4096 it exceeds 1e-12 and moved event
    // times). From t = 8192 on, the bisection runs until lo and hi are adjacent
    // doubles, which leaves hi on the first representable time at which the
    // trigger holds. The sample tolerance keeps its 1e3 margin over the
    // bisection's.
    auto bisect_tol = [&](double t) {
        const double ulp =
            std::nextafter(std::fabs(t), std::numeric_limits<double>::infinity()) - std::fabs(t);
        return std::max(BISECT_EPS, ulp);
    };
    auto sample_event_tol = [&](double t) {
        return std::max(SAMPLE_EVENT_TOL, 1e3 * bisect_tol(t));
    };
    // Enough halvings to take any finite window down to one ulp; the
    // no-progress exit below normally ends it long before.
    constexpr int BISECT_MAX_ITERS = 1100;
    // Where in (lo, hi] trigger ei stops reading `at_lo`: a rise when at_lo is
    // false, a fall (which re-arms it) when true. Returns the first time with
    // the new value, to bisection precision.
    auto bisect_trigger = [&](int ei, double lo, double hi, bool at_lo) -> double {
        // State is unchanged during the τ-step; only time advances.
        for (int iter = 0; iter < BISECT_MAX_ITERS && hi - lo > bisect_tol(hi); ++iter) {
            double mid = 0.5 * (lo + hi);
            if (mid <= lo || mid >= hi)
                break; // lo and hi are adjacent doubles: nothing left to split
            model.evaluate_functions(mid);
            const bool v = eval_ref.evaluate(events[ei].trigger_expr_idx) > 0.5;
            if (v != at_lo) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        return hi;
    };

    // ─── Two-phase t=0 trigger initialization (mirrors ODE) ──────────────────
    //
    // Seed trigger_was_true from each event's `initialValue`, evaluate the
    // actual t=0 trigger, and fire any event whose presumed-prior-state was
    // false but whose actual t=0 value is true — before the initial state
    // is recorded. After firing, re-sync trigger_was_true to the post-fire
    // truth values so subsequent transitions are detected as rising edges.
    //
    // A run that continues the previous one (issue #693) seeds them from the
    // truth that run left instead: a leg boundary is not a simulation start.
    if (n_events > 0) {
        sync_state(times.t_start);

        const NetworkModel::EventCarry *carry = model.event_carry_for(times.t_start, n_events);
        std::vector<int> t0_firing;
        t0_firing.reserve(n_events);
        for (int i = 0; i < n_events; ++i) {
            trigger_was_true[i] =
                carry != nullptr ? carry->trigger[i] != 0 : events[i].initial_value;
            double val = eval_ref.evaluate(events[i].trigger_expr_idx);
            bool now_true = (val > 0.5);
            if (now_true && !trigger_was_true[i]) {
                t0_firing.push_back(i);
            }
            trigger_was_true[i] = now_true;
        }

        // The drain leaves trigger_was_true settled, so an event that
        // falsified its own trigger can re-arm on the next rise.
        process_firing_batch(times.t_start, t0_firing);
    }

    // ─── Time-inhomogeneous propensity detection ─────────────────────────────
    //
    // Some rate laws read assignment-rule / function values that depend on
    // time() — e.g. BIOMD0000001040's synthesis rate reads the assignment rule
    // Mpl = A·exp(c·t) − B·exp(d·t). The direct method assumes propensities are
    // constant between fires, and the dependency graph only refreshes a
    // reaction's propensity when one of its *species* changes. A purely
    // time-dependent rate has no species trigger, so it would freeze at its
    // t_start value — and if every t_start propensity is zero (Mpl(0)=0 here),
    // the loop wedges in the a0==0 fast-forward and the trajectory flat-lines.
    //
    // When the model has one, switch the main loop to piecewise-constant
    // sub-stepping: cap each step at dt_max and re-evaluate all functions +
    // propensities at the new time, the same way the ODE RHS refreshes
    // assignment rules on every call. For piecewise-constant propensities the
    // discard-and-resample at the cap is exact (exponential memorylessness);
    // the only approximation is holding the rate constant over dt_max, which
    // vanishes as dt_max → 0.
    //
    // `functions_use_time()` answers this from the function expressions, which
    // is the only way to answer it. This gate used to evaluate every function
    // at t_start, the midpoint and t_end and conclude "time-invariant" when the
    // three values agreed — but a function is not pinned down by three of its
    // values, and the failure is not exotic: a rate of period 5 run over [0,10]
    // is probed at 0, 5 and 10, one whole period apart each time, so it reads
    // as constant, sub-stepping never engages, and the run reports a trajectory
    // computed against a rate that is not the model's (issue #654 — the aliased
    // horizon returned a mean 6.8× the analytic answer, silently). A syntactic
    // check cannot alias. It can over-report — a function that names `time` but
    // is constant over this window now sub-steps — and that direction costs
    // only time.
    //
    // Per reaction since issue #719: a function that reads time but feeds no
    // rate (an output, a trigger) leaves every propensity constant between
    // firings, and a Functional rate that reads no time is constant too.
    //
    // A function the caller resolved as piecewise constant in time (it reads the
    // clock only inside conditions whose crossings are this run's breakpoints)
    // leaves a rate constant between breakpoints: such a rate is static, and is
    // refreshed at each breakpoint instead of integrated (pc_rxns).
    std::vector<char> pc_fn(static_cast<std::size_t>(model.n_functions()), 0);
    if (!impl_->pc_functions.empty()) {
        const std::unordered_set<std::string> names(impl_->pc_functions.begin(),
                                                    impl_->pc_functions.end());
        const auto &fns = model.functions();
        for (std::size_t fi = 0; fi < fns.size(); ++fi)
            pc_fn[fi] = names.count(fns[fi].name) ? 1 : 0;
    }
    std::vector<char> rate_reads_time(nr, 0);
    std::vector<int> pc_rxns;
    bool time_dependent_rates = false;
    if (model.functions_use_time())
        for (int r = 0; r < nr; ++r) {
            if (reactions[r].is_rate_rule_ode || reactions[r].ode_only)
                continue;
            if (model.reaction_rate_reads_time(r, &pc_fn)) {
                rate_reads_time[r] = 1;
                time_dependent_rates = true;
            } else if (!impl_->pc_functions.empty() && model.reaction_rate_reads_time(r)) {
                pc_rxns.push_back(r);
            }
        }

    // Leave the function-bound parameters holding their t_start values, which is
    // where the probe this replaced left them. Nothing above has necessarily
    // evaluated them (the t=0 event batch does, but only when there are events),
    // and the propensity-backend setup below snapshots the live rate parameters.
    if (model.n_functions() > 0)
        model.evaluate_functions(times.t_start);

    // GH #81 — a rate-rule ODE makes every propensity that reads its target a
    // continuously-varying function of time (the target moves between fires),
    // so the piecewise-constant sub-stepping MUST engage even when no function
    // names the clock (a rate rule whose RHS reads only species does not). This
    // is also where the deterministic Euler integration of the targets is
    // driven (one Euler step per sub-step).
    if (has_rate_rules)
        time_dependent_rates = true;
    // Issues #719/#751/#753 — such a model runs on the continuous loop below.
    // Its dynamic reactions (see is_dyn) are those whose rate reads time
    // (rate_reads_time) or a rate-rule target (a reactant, a live compartment
    // volume, or anything a function reads); every other propensity is constant
    // between firings and stays in the selection tree.
    if (time_dependent_rates) {
        std::vector<char> is_cont(ns, 0);
        for (const auto &rr : rate_rule_odes)
            is_cont[rr.target0] = 1;
        std::vector<int> sup;
        for (int r = 0; r < nr; ++r) {
            const auto &rx = reactions[r];
            if (rx.is_rate_rule_ode || rx.ode_only)
                continue;
            bool d = rate_reads_time[r] != 0;
            // A rate-rule target anywhere in what the rate reads: a reactant, or
            // the live compartment volume a mass-action rate divides by. A rate
            // whose reads cannot be decided is taken to read one.
            if (!d && has_rate_rules) {
                if (!model.reaction_rate_species_support(r, sup))
                    d = true;
                else
                    for (int si : sup)
                        if (is_cont[si])
                            d = true;
            }
            if (d) {
                is_dyn[r] = 1;
                dyn.push_back(r);
            }
        }
        cont_mode = true;
        // A rate that also reads a rate-rule target is integrated, not held.
        pc_rxns.erase(
            std::remove_if(pc_rxns.begin(), pc_rxns.end(), [&](int r) { return is_dyn[r] != 0; }),
            pc_rxns.end());
    }
    const bool pc_refresh = !pc_rxns.empty();

    // ─── Decide the propensity backend + recompute-all fast loop ──────────────
    // Now that the event / rate-rule / time-dependent gates are known, choose how
    // the propensity vector is computed. The RR-style recompute-all + flat-scan
    // loop (GH #190) is eligible only for pure mass-action exact SSA with no
    // events; it is also size-gated (the O(nr) full recompute beats the
    // incremental dep-graph update only for small nr — measured win ≤44, neutral
    // by ~60). Default (no env): if the Python codegen layer handed us a
    // cc-compiled propensity .so and the model is eligible + small, load it and
    // take the fast loop — no MIR required, and production builds reach it. The
    // env overrides (PROP_CC / PROP_JIT in-process compile, RECOMPUTE_ALL) stay
    // available for ablation and bypass the size gate.
    {
        const auto on = [](const char *name) {
            const char *f = std::getenv(name);
            return f && f[0] != '\0' && f[0] != '0';
        };
        const bool want_cc = on("BNGSIM_SSA_PROP_CC");
        const bool want_mir = on("BNGSIM_SSA_PROP_JIT");
        const bool want_inprocess = want_cc || want_mir; // compile the source here
        const bool recompute_all_opt = on("BNGSIM_SSA_RECOMPUTE_ALL");
        const bool codegen_disabled = on("BNGSIM_SSA_NO_CODEGEN");

        // Backend-agnostic eligibility for the recompute-all branch.
        const bool recompute_eligible =
            !use_psa && n_events == 0 && !time_dependent_rates && !has_rate_rules && !pc_refresh;
        // Size gate for the DEFAULT path: the O(nr) full recompute beats the
        // incremental dep-graph update only for small nr (measured win ≤44,
        // neutral by ~60). BNGSIM_SSA_RECOMPUTE_ALL forces past it (ablation).
        constexpr int SSA_RECOMPUTE_DEFAULT_MAX_NR = 64;
        const bool size_ok = nr <= SSA_RECOMPUTE_DEFAULT_MAX_NR || recompute_all_opt;

        // DEFAULT (no in-process override): use the Python-provided cc .so when
        // the model is eligible and within the size gate. No MIR, reached by
        // production builds. BNGSIM_SSA_NO_CODEGEN opts out → interpreted path.
        const bool use_default =
            !want_inprocess && !codegen_disabled && recompute_eligible && size_ok;

        using PropFn = void (*)(const double *, const double *, double *);
        try {
            if (use_default && !impl_->propensity_lib_path.empty()) {
                prop_lib = std::make_unique<DynamicLibrary>(impl_->propensity_lib_path);
                prop_jit_fn = prop_lib->symbol<PropFn>("bngsim_ssa_propensities");
                a_jit.assign(nr, 0.0);
                prop_jit_active = true;
                prop_backend = "cc";
                ssa_fast_loop = true; // eligibility already established
            } else if (use_default && MirJit::available()) {
                // No cc .so — the host has no C compiler, so the Python codegen
                // layer could not build one — but this distribution embeds MIR.
                // JIT the structure-specialized propensity vector in-process so a
                // compiler-less + MIR-built install still takes the RR-parity
                // recompute-all path instead of the interpreted fallback (the
                // GH #139/#140 compiler-less role). Same eligibility + size gate
                // as the cc default; only the realization differs by a few ULP.
                auto emitted = model.emit_ssa_propensity_source_structure();
                if (emitted.second == 0) { // fully covered: every reaction mass-action
                    prop_jit = std::make_unique<MirJit>(emitted.first);
                    prop_jit_fn = prop_jit->symbol<PropFn>("bngsim_ssa_propensities");
                    a_jit.assign(nr, 0.0);
                    prop_jit_active = true;
                    prop_backend = "mir";
                    ssa_fast_loop = true; // eligibility already established
                }
            } else if (want_inprocess) {
                auto emitted = model.emit_ssa_propensity_source_structure();
                if (emitted.second == 0) { // fully covered: every reaction mass-action
                    if (want_cc) {
                        prop_cc = std::make_unique<CcJit>(emitted.first);
                        prop_jit_fn = prop_cc->symbol<PropFn>("bngsim_ssa_propensities");
                        prop_backend = "cc";
                    } else {
                        prop_jit = std::make_unique<MirJit>(emitted.first);
                        prop_jit_fn = prop_jit->symbol<PropFn>("bngsim_ssa_propensities");
                        prop_backend = "mir";
                    }
                    a_jit.assign(nr, 0.0);
                    prop_jit_active = true;
                    // Ablation: the recompute-all branch runs only when explicitly
                    // requested AND the model is eligible (size gate bypassed).
                    ssa_fast_loop = recompute_all_opt && recompute_eligible;
                }
            }
            if (prop_jit_active)
                refresh_param_vals(); // snapshot the live rate-parameter values
        } catch (const std::exception &) {
            // Any backend setup failure → interpreted compute_propensity path.
            prop_jit_active = false;
            ssa_fast_loop = false;
            prop_backend = "interpreted";
            prop_lib.reset();
            prop_cc.reset();
            prop_jit.reset();
            prop_jit_fn = nullptr;
        }
    }

    // ─── Record initial state (post-t=0-events) ──────────────────────────────
    model.update_observables(conc.data());
    model.evaluate_functions(times.t_start);
    for (int j = 0; j < n_obs; ++j) {
        obs_buf[j] = model.observables()[j].total;
    }
    result.record(0, times.t_start, conc.data(), obs_buf.data());
    if (rec_stats)
        rs_record(0, times.t_start);
    if (n_func > 0) {
        auto fvals = model.function_values();
        result.record_expressions(0, fvals.data());
    }

    // ─── Initial propensity computation (all reactions) ──────────────────────
    recompute_all_propensities();

    // ─── Main simulation loop ────────────────────────────────────────────────

    double t = times.t_start;
    int next_output = 1;
    long total_steps = 0;

    // Times at which a rate or a trigger may jump: those the caller resolved
    // (set_breakpoints, from the model's time conditions) and the knots of every
    // time-indexed table function, whose value or slope breaks there. Sorted and
    // unique; read by both loops.
    std::vector<double> bps = impl_->breakpoints;
    for (double x : model.time_table_knots())
        bps.push_back(x);
    std::sort(bps.begin(), bps.end());
    bps.erase(std::unique(bps.begin(), bps.end()), bps.end());

    // The resolution at which a trigger that moves with time is watched where
    // nothing else bounds it: no gap between two looks exceeds this, which is
    // the frozen-rate sub-step the loop used to take.
    const double ev_dt = (times.t_end - times.t_start) / 1000.0;

    // The first time in (t_lo, t_hi] at which any trigger changes value, with
    // the discrete state held where it is (∞ if none): a rise is an event, a
    // fall re-arms the trigger (fire_rising_edges records both). Each trigger
    // is looked at on every breakpoint in the window, between each pair of them
    // (a trigger such as `time > 37.3 && time < 37.5` is false on both edges of
    // its window), and on a grid no coarser than ev_dt; the first interval in
    // which any trigger differs from its recorded truth is bisected. Probing only
    // the window's end saw neither a trigger true between two looks nor one that
    // fell and rose again, so a periodic trigger fired once.
    // Only a trigger that reads the clock can change while the state holds; the
    // rest change at a firing, where fire_rising_edges looks at every trigger.
    std::vector<int> time_triggers;
    for (int ei = 0; ei < n_events; ++ei)
        if (model.event_trigger_reads_time(ei))
            time_triggers.push_back(ei);
    std::vector<char> probe_prev, probe_cur; // the continuous loop's scan
    auto probe_events_in_window = [&](double t_lo, double t_hi) -> double {
        double t_event = std::numeric_limits<double>::infinity();
        // Nothing past the run's end can fire in it. The discrete loop asks up
        // to its next firing, which with a tiny total propensity lies far past
        // t_end: the grid then ran to it (millions of looks, past the timeout,
        // and past INT_MAX in the look count).
        t_hi = std::min(t_hi, times.t_end);
        if (time_triggers.empty() || !(t_hi > t_lo))
            return t_event;
        // Until a trigger changes, its recorded truth is what each look compares
        // with, so nothing is copied or stored per window.
        bool first = true;
        double t_prev = t_lo;
        const auto look = [&](double ts) -> bool {
            if (first)
                sync_state(ts);
            else
                model.evaluate_functions(ts);
            first = false;
            bool changed = false;
            for (int ei : time_triggers) {
                const bool was = trigger_was_true[ei];
                if ((eval_ref.evaluate(events[ei].trigger_expr_idx) > 0.5) == was)
                    continue;
                changed = true;
                t_event = std::min(t_event, bisect_trigger(ei, t_prev, ts, was));
                model.evaluate_functions(ts); // the bisection moved the functions' time
            }
            t_prev = ts;
            return changed;
        };
        // One window segment: its interior on a grid no coarser than ev_dt, at
        // least one interior look between two breakpoints, then its end.
        const auto segment = [&](double a, double b, bool both_bp) -> bool {
            const int n = std::max(both_bp ? 2 : 1, static_cast<int>(std::ceil((b - a) / ev_dt)));
            for (int k = 1; k < n; ++k)
                if (look(a + (b - a) * k / n))
                    return true;
            return look(b);
        };
        double a = t_lo;
        bool a_bp = false;
        if (!bps.empty()) {
            a_bp = std::binary_search(bps.begin(), bps.end(), t_lo);
            for (auto it = std::upper_bound(bps.begin(), bps.end(), t_lo);
                 it != bps.end() && *it < t_hi; ++it) {
                if (segment(a, *it, a_bp))
                    return t_event;
                a = *it;
                a_bp = true;
            }
        }
        segment(a, t_hi, a_bp && std::binary_search(bps.begin(), bps.end(), t_hi));
        // The model is left synced at the last time looked at. Nothing reads it
        // as t_lo's: a rate here reads no clock (or the model would be on the
        // continuous loop), and the trigger sweep and every event path sync
        // first.
        return t_event;
    };

    // Wall-clock check is hoisted out of the per-reaction hot loop with a
    // stride so the cost amortizes to a few ns/step. The stride is small
    // enough that responsiveness stays sub-100µs for typical propensity
    // densities.
    constexpr long TIMEOUT_CHECK_STRIDE = 1024;
    long steps_since_timeout_check = 0;

    // Reused event-firing batch buffer (T5): the per-firing / per-event-window
    // rising-edge sweeps below cleared-and-refilled this instead of heap-
    // allocating a fresh std::vector<int> each time. Byte-identical — same
    // contents, reused storage. Each use site aliases it as `firing`.
    std::vector<int> firing_scratch;

    // Fire, as one §4.11.6 batch, every event whose trigger is true now but
    // was not when trigger_was_true last recorded it, and leave trigger_was_true
    // holding the settled truth of every trigger. Every caller after t_start
    // goes through here. The two time-event callers used to take the batch
    // only from the events their probe had located, then record every
    // trigger's current truth; an event that rose some other way in the window
    // (a trigger on a rate-rule target, which the probe reads frozen) was
    // marked as already true and never fired, the issue #761 mechanism on
    // another path.
    //
    // It syncs the state first. A trigger reads the species, observables and
    // functions the model holds, and nothing else had put the post-firing
    // state there: after a firing the sweep read the state of the last sync,
    // and a state trigger (`S >= 5`) was caught only by the next window's
    // probe, which synced at the window's end. That probe now looks only at
    // triggers that read the clock (issue #719).
    auto fire_rising_edges = [&](double t_now) -> bool {
        sync_state(t_now);
        firing_scratch.clear();
        for (int ei = 0; ei < n_events; ++ei) {
            const bool now_true = eval_ref.evaluate(events[ei].trigger_expr_idx) > 0.5;
            if (now_true && !trigger_was_true[ei])
                firing_scratch.push_back(ei); // the drain marks it true
            else
                trigger_was_true[ei] = now_true;
        }
        return process_firing_batch(t_now, firing_scratch);
    };

    // Fire reaction `selected` once: its count, its species (PSA: a leap of m_r
    // molecules), the reverse-fire diagnostic. Shared by both loops.
    auto apply_firing = [&](int selected) {
        if (rec_stats)
            rs_count[selected] += 1.0;

        // 7. Execute reaction: update species populations
        //    PSA: scale stoichiometric coefficients by 1/λ_r
        //    Per-species volume_factor: SBML loader stores values as
        //    `amount/V_c`, so each ±1 amount fire is `±1/V_c` in storage
        //    units, taken on the count by fire_species (issue #692). Default
        //    volume_factor=1.0 → identical to ±1 fires.
        const auto &rxn = reactions[selected];
        double stoich_scale = 1.0;
        if (use_psa) {
            // The leap size m_r itself: 1/(1/m) is not m for 11,867 of the
            // integers below 1e5 (1/(1/98) = 98.00000000000001), which left a
            // count off a whole number after a leap.
            stoich_scale = psa_m_at[selected];
        }

        // GH #110 — sign-split firing, no non-negativity floor.
        //   dir > 0 (rate >= 0): reactants consumed, products produced (normal).
        //   dir < 0 (rate  < 0): reactants produced, products consumed — the
        //     reaction runs in reverse with propensity |rate|, exactly as the
        //     CVODE path integrates a negative rate (derivs[reactant] -= rate
        //     grows the reactant). Both directions apply the full ±step to BOTH
        //     sides, so mass is conserved and the SSA mean tracks the ODE.
        // Species counts are NOT clamped at zero: a count goes negative exactly
        // as the literal rate law dictates (matching CVODE, which has no
        // CVodeSetConstraints). Non-negativity is the modeler's job. Each
        // downward zero-crossing is recorded for the run diagnostic.
        const int dir = rxn_dir[selected];
        if (dir < 0) {
            ++reverse_fire_count;
            if (first_reverse_rxn < 0)
                first_reverse_rxn = selected;
        }
        const double rstep = dir * stoich_scale;

        for (int ri : rxn.reactant_indices) {
            int si = ri - 1; // 1-based → 0-based
            if (si >= 0 && si < ns && !species_list[si].fixed) {
                fire_species(si, -rstep);
            }
        }
        for (int pi : rxn.product_indices) {
            int si = pi - 1;
            if (si >= 0 && si < ns && !species_list[si].fixed) {
                fire_species(si, rstep);
            }
        }
    };
    // After a firing at the current t: refresh the propensities it affects.
    auto refresh_after_firing = [&](int selected) {
        // 9. Update propensities for AFFECTED reactions only: O(k log N)
        //    - If model has functional rate laws, update observables + functions first
        //    - Use the dependency graph's precomputed affected-reaction set (GH #190)
        //    - Recompute their propensities and update the selection structure

        if (dep_graph.has_functional_rates) {
            model.update_observables(conc.data());
            model.evaluate_functions(t);
        }

        // GH #149: refill the JIT'd propensity buffer from the post-fire conc[]
        // before the affected reactions read it (no-op when the fast path is off).
        refresh_jit_propensities();

        for (int r : dep_graph.affected_reactions(selected))
            set_propensity(r);
    };

    // A rate that reads the clock only through piecewise-constant functions
    // jumps only at a breakpoint. It is re-read once the loop's time has reached
    // a breakpoint, by whatever path got it there (the stop at the breakpoint, a
    // firing or an event landing on it, an idle stretch), at the top of each
    // loop: pc_seen counts the breakpoints at or before the last re-read.
    std::size_t pc_seen = 0;
    // Only for a run with held rates: pc_seen moves in pc_refresh_at alone.
    auto pc_next = [&]() {
        return pc_seen < bps.size() ? bps[pc_seen] : std::numeric_limits<double>::infinity();
    };
    auto pc_due = [&](double t_now) { return pc_refresh && pc_next() <= t_now; };
    // Read inside the interval the rate is constant on, (t_now, next breakpoint),
    // not at t_now: at a breakpoint a condition is still on its old side (`time()
    // > 5` is false at 5), and the rate it switches holds from just after.
    auto pc_refresh_at = [&](double t_now) {
        while (pc_seen < bps.size() && bps[pc_seen] <= t_now)
            ++pc_seen;
        const double t_hi = std::min(pc_next(), times.t_end);
        const double t_in = t_hi > t_now ? 0.5 * (t_now + t_hi) : t_now;
        sync_state(t_in);
        refresh_jit_propensities();
        for (int r : pc_rxns)
            set_propensity(r);
        model.evaluate_functions(t_now);
    };
    // A guard at every output row that such a rate has not moved since it was
    // set: a classification that let a moving rate through would otherwise
    // freeze it silently, so the run is refused, loudly. Called with the state
    // and the functions at t_row. Not within a few ulps of a breakpoint, t_start
    // or t_end, where a condition may read its boundary value (`time() >= 5` at
    // 5) or the crossing Python placed may sit an ulp from the one the model
    // evaluates; and to a relative 1e-10, the compiled and interpreted rate
    // laws rounding differently.
    auto pc_guard = [&](double t_row) {
        if (!pc_refresh)
            return;
        const double tol =
            64.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::fabs(t_row));
        if (std::fabs(t_row - times.t_start) <= tol || std::fabs(times.t_end - t_row) <= tol)
            return;
        const auto it = std::lower_bound(bps.begin(), bps.end(), t_row - tol);
        if (it != bps.end() && *it <= t_row + tol)
            return;
        for (int r : pc_rxns) {
            const double v = std::fabs(model.compute_propensity(r, conc.data()));
            const double p = propensities[r];
            if (std::fabs(v - p) <= 1e-10 * std::max(v, p))
                continue;
            char buf[96];
            std::snprintf(buf, sizeof buf, " at t = %.17g (%.17g, set as %.17g)", t_row, v, p);
            throw std::runtime_error(
                std::string(use_psa ? "PSA" : "SSA") + ": the rate of reaction " +
                reaction_label(model, reactions[r]) + " moved between breakpoints" + buf +
                ", though its time dependence was classified piecewise constant. This is "
                "a bngsim bug; please report it.");
        }
    };

    // ─── GH #190: RR-style recompute-all + flat-scan fast loop ────────────────
    //
    // Engaged (ssa_fast_loop, decided above) when a value-specialized propensity
    // vector is available and the model is recompute-all eligible. The whole
    // vector is refilled with ONE native call, which makes the dependency-graph
    // incremental machinery (the affected-set lookup + per-affected
    // set_propensity + sum-tree update that the main loop below runs every step)
    // pure overhead. RoadRunner's direct method instead recomputes every
    // propensity each step and flat-scans a single contiguous cumulative pass
    // that both sums (a0) and selects. This branch mirrors that: one refill, one
    // flat pass for a0 + sign, one flat scan for selection — no dependency graph,
    // no sum tree, and no per-step model.set_current_time (time() is unread for
    // pure mass-action, so the main loop's out-of-line call is exactly the
    // bookkeeping this branch sheds; the final write-back still publishes t). It
    // is bit-identical to the incremental flat+JIT realization: the same
    // |a_jit[r]| values, the same index-order sum, the same index-order scan, and
    // the same RNG draw order (one r1 for τ, one r2 for selection).
    if (ssa_fast_loop) {
        while (next_output < n_out) {
            if (budget.active() && ++steps_since_timeout_check >= TIMEOUT_CHECK_STRIDE) {
                budget.check();
                steps_since_timeout_check = 0;
            }

            // One native call refills the entire propensity vector from conc[]
            // (param_vals is constant across this events-free fast loop).
            prop_jit_fn(conc.data(), param_vals.data(), a_jit.data());
            if (rec_stats) { // GH #616 — the refilled vector is in force from t on
                for (int r = 0; r < nr; ++r)
                    rs_bank(r, t, (a_jit[r] < 0.0) ? -a_jit[r] : a_jit[r]);
            }

            // Single contiguous pass: total propensity a0 over |a_jit| plus each
            // reaction's firing direction (GH #110 sign-split — store |rate| for
            // selection, fire in reverse when the literal rate law is negative).
            double a0 = 0.0;
            for (int r = 0; r < nr; ++r) {
                double sp = a_jit[r];
                if (sp < 0.0) {
                    rxn_dir[r] = -1;
                    a0 -= sp;
                } else {
                    rxn_dir[r] = 1;
                    a0 += sp;
                }
            }
            if (!std::isfinite(a0)) { // issue #809: NaN fails every comparison below
                for (int r = 0; r < nr; ++r)
                    if (!std::isfinite(a_jit[r]))
                        refuse_nonfinite_propensity(r, a_jit[r], t);
                refuse_nonfinite_propensity(-1, a0, t);
            }

            // Stuck — fast-forward all remaining samples at the frozen state.
            if (a0 <= 0.0) {
                model.update_observables(conc.data());
                for (int j = 0; j < n_obs; ++j)
                    obs_buf[j] = model.observables()[j].total;
                while (next_output < n_out) {
                    result.record(next_output, t_out[next_output], conc.data(), obs_buf.data());
                    if (rec_stats)
                        rs_record(next_output, t_out[next_output]);
                    // Issue #569 — the frozen state still has function values, and
                    // every other recording site in this loop pairs record() with
                    // them. Leaving these rows untouched is not "unrecorded":
                    // set_expression_names zero-fills expressions_, so they read
                    // back as a perfectly plausible 0.0 with no sentinel.
                    if (n_func > 0) {
                        model.evaluate_functions(t_out[next_output]);
                        auto fvals = model.function_values();
                        result.record_expressions(next_output, fvals.data());
                    }
                    ++next_output;
                }
                break;
            }

            // Time to next reaction: tau = -ln(r1) / a0.
            double r1 = next_u01();
            while (r1 == 0.0)
                r1 = next_u01();
            double tau = -std::log(r1) / a0;
            double t_proposed = t + tau;

            // Record output points strictly before the fire time. Evaluated per
            // sample, matching the main loop byte-for-byte. `time_dependent_rates`
            // gates this branch off, and since #654 that gate is the syntactic
            // `functions_use_time()` — so the functions reaching here really are
            // time-invariant and a single pre-loop evaluation would do. Per
            // sample anyway: it is the same answer at a cost this branch does not
            // notice (it runs once per output row, not once per fire), and it is
            // the one form that stays correct if the gate ever widens again.
            while (next_output < n_out && t_proposed >= t_out[next_output]) {
                model.update_observables(conc.data());
                for (int j = 0; j < n_obs; ++j)
                    obs_buf[j] = model.observables()[j].total;
                result.record(next_output, t_out[next_output], conc.data(), obs_buf.data());
                if (rec_stats)
                    rs_record(next_output, t_out[next_output]);
                if (n_func > 0) {
                    model.evaluate_functions(t_out[next_output]);
                    auto fvals = model.function_values();
                    result.record_expressions(next_output, fvals.data());
                }
                ++next_output;
            }
            if (next_output >= n_out)
                break;

            // Select the reaction: flat cumulative scan in index order (the same
            // selection semantics as the tree descent, contiguous + L1-hot).
            // Strict, as sel_find is: a weight-0 reaction is never taken.
            double target = next_u01() * a0;
            double cum = 0.0;
            int selected = -1;
            int last_positive = nr - 1;
            for (int r = 0; r < nr; ++r) {
                double sp = a_jit[r];
                sp = (sp < 0.0) ? -sp : sp;
                cum += sp;
                if (sp > 0.0)
                    last_positive = r;
                if (cum > target) {
                    selected = r;
                    break;
                }
            }
            if (selected < 0)
                selected = last_positive;

            if (rec_stats)
                rs_count[selected] += 1.0;

            // Execute (GH #110 sign-split firing, no non-negativity floor;
            // stoich_scale is 1 because PSA is gated out of this path).
            const auto &rxn = reactions[selected];
            const int dir = rxn_dir[selected];
            if (dir < 0) {
                ++reverse_fire_count;
                if (first_reverse_rxn < 0)
                    first_reverse_rxn = selected;
            }
            const double rstep = dir;
            for (int ri : rxn.reactant_indices) {
                int si = ri - 1; // 1-based → 0-based
                if (si >= 0 && si < ns && !species_list[si].fixed)
                    fire_species(si, -rstep);
            }
            for (int pi : rxn.product_indices) {
                int si = pi - 1;
                if (si >= 0 && si < ns && !species_list[si].fixed)
                    fire_species(si, rstep);
            }

            // Advance time only (see header note: no per-step set_current_time).
            t = t_proposed;
            ++total_steps;
        }
    }

    // ─── The continuous loop: time-dependent rates and rate rules ─────────────
    //
    // Issues #719, #751, #753. Between two firings the discrete state is fixed,
    // but a model with time-dependent rates or rate rules still moves: every
    // dynamic propensity (see is_dyn) follows t and the rate-rule targets y,
    // and y follows its ODE. The next firing is then at the time τ where the
    // integrated hazard
    //     H(τ) = A_s·(τ − t) + ∫_t^τ a_d(s, y(s)) ds
    // reaches an Exp(1) draw E, A_s being the constant sum the tree holds and
    // a_d the sum of the dynamic propensities; which reaction fires is chosen
    // from the propensities at τ. That is exact for the time-inhomogeneous
    // process (the time-change theorem), where the old loop held every rate
    // fixed over windows of horizon/1000 and moved y by forward Euler: results
    // depended on the run's length (2-6x off, #719), an idle jump to an event
    // froze the rates and took one Euler step over the whole gap (x' = -x went
    // to -8, #751), and a stiff rate rule blew up (#753).
    //
    // The loop steps (y, ∫a_d) together over panels with error control:
    //   - y by an L-stable Rosenbrock method (Shampine & Reichelt's ode23s,
    //     order 2 with a third-order error estimate and dense output), with a
    //     finite-difference Jacobian. Being implicit it takes stiff rate rules
    //     in its stride; being one-step it restarts after a firing at no cost.
    //   - ∫a_d by Simpson's rule over the panel, error-controlled against the
    //     midpoint rule, with a_d evaluated on the dense y. Inside the panel the
    //     quadratic through a_d's three values gives H as a cubic, on which τ
    //     is found; an event trigger (which may read y) is checked at the
    //     panel's end and located on the dense y.
    // A panel never crosses a breakpoint (a time at which a rate jumps, from
    // set_breakpoints) or t_end. After a firing or an event the discrete state
    // has moved, and the next panel starts from there.
    //
    // Tolerances: rate-rule targets rtol 1e-6, atol 1e-9; the hazard integral
    // 1e-7 + 1e-6·|∫| per panel, so a firing time is off by far less than the
    // spread of the draw that places it.
    //
    // A piecewise-constant rate was set at t_start, where a condition reads its
    // boundary value (`time() > 0` at 0); it holds the first interval's value.
    if (pc_refresh)
        pc_refresh_at(t);
    if (time_dependent_rates) {
        const int m = static_cast<int>(rate_rule_odes.size());
        const int nd = static_cast<int>(dyn.size());
        constexpr double RTOL_Y = 1e-6, ATOL_Y = 1e-9;
        constexpr double RTOL_H = 1e-6, ATOL_H = 1e-7;
        const double dR = 1.0 / (2.0 + std::sqrt(2.0)); // ode23s
        const double e32 = 6.0 + std::sqrt(2.0);
        const bool need_dyn_nodes = rec_stats || use_psa;

        // A firing that changes no species a dynamic propensity reads leaves the
        // panel's a_d polynomial valid past it: the loop carries on in the same
        // panel from the firing, with a fresh draw, instead of evaluating a_d at
        // five new nodes. A rate that reads only time (a forcing, a dosing
        // schedule) keeps its panel through every firing. Only without rate
        // rules, whose trajectory a firing redirects. Decided once, per reaction,
        // from what each dynamic propensity reads; anything that cannot be
        // decided (a table function, a species a rule sets) keeps no panel.
        std::vector<char> fire_keeps_panel(nr, 0);
        if (m == 0) {
            std::vector<char> read(ns, 0);
            bool all = false;
            std::vector<int> sup;
            for (int r : dyn) {
                if (!model.reaction_rate_species_support(r, sup)) {
                    all = true;
                    break;
                }
                for (int si : sup) {
                    if (species_list[si].continuous)
                        all = true;
                    read[si] = 1;
                }
                if (use_psa) // the leap m_r reads the products' populations too
                    for (int pi : reactions[r].product_indices)
                        if (pi >= 1 && pi <= ns)
                            read[pi - 1] = 1;
                if (all)
                    break;
            }
            if (!all)
                for (int r = 0; r < nr; ++r) {
                    char keeps = 1;
                    for (int si : dep_graph.reaction_to_affected_species[r])
                        if (read[si])
                            keeps = 0;
                    fire_keeps_panel[r] = keeps;
                }
        }
        // ...and whether that firing's refresh can skip the observables and
        // functions: only when no static propensity it moves reads a function
        // (a Functional rate that reads no time is static, and reads them).
        std::vector<char> lean_ok(nr, 1);
        {
            std::vector<char> static_reads_fn(nr, 0);
            for (int r = 0; r < nr; ++r)
                static_reads_fn[r] =
                    !is_dyn[r] && (reactions[r].rate_law_type == RateLawType::Functional ||
                                   model.reaction_rate_reads_functions(r));
            for (int r = 0; r < nr; ++r)
                for (int a : dep_graph.affected_reactions(r))
                    if (static_reads_fn[a])
                        lean_ok[r] = 0;
        }

        std::vector<double> y(m), y1(m), ytmp(m), F0(m), F1(m), F2(m), Tt(m);
        std::vector<double> k1(m), k2(m), k3(m), J(static_cast<std::size_t>(m) * m);
        std::vector<double> W(static_cast<std::size_t>(m) * m);
        std::vector<int> piv(m);
        // The panel's five nodes: Gauss–Lobatto, θ = 0, ½ ∓ √(3/7)/2, ½, 1. Equally
        // spaced nodes aliased a rate periodic in their spacing: at its cap of
        // ev_dt a panel's quarter points are whole periods of a rate of period
        // ev_dt/4, which read as constant (E[N] came out twice the truth, z = 106).
        // The Lobatto gaps are incommensurate, so no period lines up with them.
        const double NODE[5] = {0.0, 0.5 - 0.5 * std::sqrt(3.0 / 7.0), 0.5,
                                0.5 + 0.5 * std::sqrt(3.0 / 7.0), 1.0};
        // Dynamic propensities at those nodes.
        std::vector<std::vector<double>> dnv(5, std::vector<double>(need_dyn_nodes ? nd : 0));
        std::vector<double> dnow(nd);
        for (int i = 0; i < m; ++i)
            y[i] = conc[rate_rule_odes[i].target0];

        // The model's observables hold the current discrete state (cleared by
        // anything that moves it outside a sync).
        bool obs_current = false;
        // Put a rate-rule state into conc[] (and the count array beside it).
        auto load_y = [&](const double *yv) {
            for (int i = 0; i < m; ++i) {
                const int x = rate_rule_odes[i].target0;
                conc[x] = yv[i];
                counts[x] = conc[x] * rate_rule_odes[i].vf;
            }
        };
        // The continuous right-hand side at (s, yv): the rate-rule derivatives
        // into fout (when non-null) and the dynamic propensities (each |a| into
        // raw when non-null); returns their effective sum (PSA: each over m_r).
        auto eval_cont = [&](double s, const double *yv, double *fout, double *raw, bool want_dyn,
                             bool synced = false) -> double {
            if (!synced) {
                if (m > 0 || !obs_current) {
                    load_y(yv);
                    sync_state(s);
                    obs_current = true;
                } else {
                    // No rate rule: across a panel only time moves, so the
                    // observables already hold this state; only the functions
                    // of time need evaluating.
                    model.evaluate_functions(s);
                }
            }
            if (fout) {
                for (int i = 0; i < m; ++i) {
                    const auto &rr = rate_rule_odes[i];
                    const double f = model.compute_propensity(rr.rxn, conc.data());
                    if (!std::isfinite(f))
                        throw std::runtime_error(
                            std::string(use_psa ? "PSA" : "SSA") + ": the rate rule for " +
                            model.species()[rr.target0].name + " evaluates to " +
                            (std::isnan(f) ? "nan" : (f > 0 ? "inf" : "-inf")) +
                            " at t=" + std::to_string(s));
                    fout[i] = rr.per_species ? f / rr.vf : f;
                }
            }
            double sum = 0.0;
            if (!want_dyn)
                return sum;
            for (int k = 0; k < nd; ++k) {
                const int r = dyn[k];
                const double v = model.compute_propensity(r, conc.data());
                if (!std::isfinite(v))
                    refuse_nonfinite_propensity(r, v, s);
                const double a = v < 0.0 ? -v : v;
                if (raw)
                    raw[k] = a;
                sum += use_psa ? a * scaling_factors[r] : a;
            }
            return sum;
        };

        // Dense LU of the small W = I − h·d·J (partial pivoting); false if singular.
        auto lu = [&]() -> bool {
            for (int c = 0; c < m; ++c) {
                int p = c;
                double best = std::fabs(W[static_cast<std::size_t>(c) * m + c]);
                for (int r2 = c + 1; r2 < m; ++r2) {
                    const double v = std::fabs(W[static_cast<std::size_t>(r2) * m + c]);
                    if (v > best) {
                        best = v;
                        p = r2;
                    }
                }
                if (!(best > 0.0))
                    return false;
                piv[c] = p;
                if (p != c)
                    for (int j = 0; j < m; ++j)
                        std::swap(W[static_cast<std::size_t>(c) * m + j],
                                  W[static_cast<std::size_t>(p) * m + j]);
                const double d = W[static_cast<std::size_t>(c) * m + c];
                for (int r2 = c + 1; r2 < m; ++r2) {
                    double &l = W[static_cast<std::size_t>(r2) * m + c];
                    l /= d;
                    for (int j = c + 1; j < m; ++j)
                        W[static_cast<std::size_t>(r2) * m + j] -=
                            l * W[static_cast<std::size_t>(c) * m + j];
                }
            }
            return true;
        };
        auto solve = [&](std::vector<double> &b) {
            for (int c = 0; c < m; ++c)
                if (piv[c] != c)
                    std::swap(b[c], b[piv[c]]);
            for (int r2 = 1; r2 < m; ++r2)
                for (int j = 0; j < r2; ++j)
                    b[r2] -= W[static_cast<std::size_t>(r2) * m + j] * b[j];
            for (int r2 = m - 1; r2 >= 0; --r2) {
                for (int j = r2 + 1; j < m; ++j)
                    b[r2] -= W[static_cast<std::size_t>(r2) * m + j] * b[j];
                b[r2] /= W[static_cast<std::size_t>(r2) * m + r2];
            }
        };

        // Panel state: [t, t + hh], a_d at its start, middle and end (A0, A1,
        // A2), the Rosenbrock stages k1, k2 for the dense y.
        double Av[5] = {0.0, 0.0, 0.0, 0.0, 0.0}, hh = 0.0;
        double &A0 = Av[0];
        // No panel longer than ev_dt: a stretch where the rate reads 0 at every
        // node has no error to measure, and a step left to grow 5× a panel would
        // walk over a bump or pulse between two nodes.
        const double hmax = ev_dt;
        double h = hmax;
        bool fresh = true;   // F0/A0 must be evaluated at (t, y)
        bool synced = false; // ...and the model already holds the state at t
        bool jac_ok = false; // J and ∂f/∂t are for the current panel start
        std::size_t next_bp = 0;

        auto dense_y = [&](double th, double *out) {
            const double c1 = th * (1.0 - th) / (1.0 - 2.0 * dR);
            const double c2 = th * (th - 2.0 * dR) / (1.0 - 2.0 * dR);
            for (int i = 0; i < m; ++i)
                out[i] = y[i] + hh * (c1 * k1[i] + c2 * k2[i]);
        };
        // The quartic through five values at the panel's nodes in the monomial
        // basis: c = V⁻¹ v, V[i][k] = θ_i^k, inverted once here.
        double Vinv[5][5];
        {
            double V[5][10];
            for (int i = 0; i < 5; ++i)
                for (int k = 0; k < 10; ++k)
                    V[i][k] = k < 5 ? std::pow(NODE[i], k) : (k - 5 == i ? 1.0 : 0.0);
            for (int c = 0; c < 5; ++c) {
                int p = c;
                for (int r2 = c + 1; r2 < 5; ++r2)
                    if (std::fabs(V[r2][c]) > std::fabs(V[p][c]))
                        p = r2;
                for (int k = 0; k < 10; ++k)
                    std::swap(V[c][k], V[p][k]);
                const double d = V[c][c];
                for (int k = 0; k < 10; ++k)
                    V[c][k] /= d;
                for (int r2 = 0; r2 < 5; ++r2)
                    if (r2 != c) {
                        const double f = V[r2][c];
                        for (int k = 0; k < 10; ++k)
                            V[r2][k] -= f * V[c][k];
                    }
            }
            for (int i = 0; i < 5; ++i)
                for (int j = 0; j < 5; ++j)
                    Vinv[i][j] = V[i][5 + j];
        }
        auto coeffs = [&](const double v[5], double c[5]) {
            for (int k = 0; k < 5; ++k) {
                c[k] = 0.0;
                for (int j = 0; j < 5; ++j)
                    c[k] += Vinv[k][j] * v[j];
            }
        };
        // ∫_0^θ of the quartic with coefficients c, in time units.
        auto poly_int = [&](const double c[5], double th) {
            double acc = 0.0;
            for (int k = 4; k >= 0; --k)
                acc = acc * th + c[k] / (k + 1);
            return hh * th * acc;
        };
        double Ac[5] = {0.0, 0.0, 0.0, 0.0, 0.0}; // the panel's a_d quartic
        auto L_at = [&](double th) { return poly_int(Ac, th); };
        auto quad_int = [&](int k, double th) { // one dynamic reaction's integral
            const double v[5] = {dnv[0][k], dnv[1][k], dnv[2][k], dnv[3][k], dnv[4][k]};
            double c[5];
            coeffs(v, c);
            return poly_int(c, th);
        };

        // Dynamic reactions' integrated propensities (stats, PSA): banked up to
        // bank_th of the current panel.
        double bank_th = 0.0;
        auto bank_dyn = [&](double th) {
            if (!need_dyn_nodes || th <= bank_th)
                return;
            for (int k = 0; k < nd; ++k) {
                const int r = dyn[k];
                const double I = quad_int(k, th) - quad_int(k, bank_th);
                if (rec_stats)
                    rs_integral[r] += I;
                if (use_psa) {
                    psa_exact_int += I;
                    psa_scaled_int += I * scaling_factors[r];
                    psa_qexc_int[r] += (psa_m_at[r] - 1.0) * I;
                }
            }
            bank_th = th;
        };

        // Record output row next_output at time tk = t + th·hh (inside the panel).
        auto record_at = [&](double tk, double th) {
            if (m > 0) {
                dense_y(th, ytmp.data());
                load_y(ytmp.data());
            }
            sync_state(tk);
            pc_guard(tk);
            for (int j = 0; j < n_obs; ++j)
                obs_buf[j] = model.observables()[j].total;
            result.record(next_output, tk, conc.data(), obs_buf.data());
            if (rec_stats) {
                for (int r = 0; r < nr; ++r)
                    rs_scratch[r] = rs_integral[r] + rs_a[r] * (tk - rs_last_t[r]);
                if (need_dyn_nodes)
                    for (int k = 0; k < nd; ++k)
                        rs_scratch[dyn[k]] += quad_int(k, th) - quad_int(k, bank_th);
                result.record_reaction_stats(next_output, rs_count.data(), rs_scratch.data());
            }
            if (n_func > 0) {
                auto fvals = model.function_values();
                result.record_expressions(next_output, fvals.data());
            }
            ++next_output;
        };

        auto draw_exp = [&]() {
            double r1 = next_u01();
            while (r1 == 0.0)
                r1 = next_u01();
            return -std::log(r1);
        };
        double E = draw_exp();
        double Hz = 0.0; // hazard accumulated since the last firing
        // The panel is [tp, s1p], hh long; the loop is at t = tp + th0·hh inside
        // it (th0 = 0 but after a firing that kept the panel). h_next_p is the
        // step the panel's error test proposed for the next one.
        double tp = t, th0 = 0.0, s1p = t, h_next_p = h;
        bool carry = false;
        constexpr int MAX_FLOOR_STEPS = 100;
        int floor_steps = 0; // panels accepted at hmin in a row
        constexpr long MAX_TINY_PANELS = 1000000;
        long tiny_panels = 0;

        while (t < times.t_end) {
            if (budget.active() && ++steps_since_timeout_check >= TIMEOUT_CHECK_STRIDE) {
                budget.check();
                steps_since_timeout_check = 0;
            }
            if (pc_due(t)) {
                // A new panel from here, on the re-read rates.
                pc_refresh_at(t);
                carry = false;
                fresh = true;
                synced = false;
            }
            while (next_bp < bps.size() && bps[next_bp] <= t)
                ++next_bp;
            const double t_stop =
                (next_bp < bps.size() && bps[next_bp] < times.t_end) ? bps[next_bp] : times.t_end;

            const double hmin =
                64.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::fabs(t));
            if (!carry) {
                if (fresh) {
                    A0 = eval_cont(t, y.data(), F0.data(), need_dyn_nodes ? dnv[0].data() : nullptr,
                                   true, synced);
                    fresh = false;
                    synced = false;
                    jac_ok = false;
                }
                if (m > 0 && !jac_ok) {
                    // Finite-difference Jacobian ∂f/∂y and ∂f/∂t at the panel start.
                    const double sq = std::sqrt(std::numeric_limits<double>::epsilon());
                    for (int j = 0; j < m; ++j) {
                        ytmp = y;
                        const double dj = sq * std::max(std::fabs(y[j]), 1e-3);
                        ytmp[j] += dj;
                        eval_cont(t, ytmp.data(), F1.data(), nullptr, false);
                        for (int i = 0; i < m; ++i)
                            J[static_cast<std::size_t>(i) * m + j] = (F1[i] - F0[i]) / dj;
                    }
                    const double dt = sq * std::max(std::fabs(t), std::max(h, 1e-8));
                    eval_cont(t + dt, y.data(), Tt.data(), nullptr, false);
                    for (int i = 0; i < m; ++i)
                        Tt[i] = (Tt[i] - F0[i]) / dt;
                    jac_ok = true;
                }

                // One error-controlled panel.
                double h_next = h;
                h = std::min(h, hmax);
                while (true) {
                    const bool to_stop = h >= t_stop - t;
                    hh = to_stop ? t_stop - t : h;
                    const double s1 = to_stop ? t_stop : t + hh;
                    double err = 0.0;
                    if (m > 0) {
                        for (int i = 0; i < m; ++i)
                            for (int j = 0; j < m; ++j)
                                W[static_cast<std::size_t>(i) * m + j] =
                                    (i == j ? 1.0 : 0.0) -
                                    hh * dR * J[static_cast<std::size_t>(i) * m + j];
                        if (!lu()) {
                            err = 1e10; // singular: shrink
                        } else {
                            for (int i = 0; i < m; ++i)
                                k1[i] = F0[i] + hh * dR * Tt[i];
                            solve(k1);
                            for (int i = 0; i < m; ++i)
                                ytmp[i] = y[i] + 0.5 * hh * k1[i];
                            eval_cont(t + 0.5 * hh, ytmp.data(), F1.data(), nullptr, false);
                            for (int i = 0; i < m; ++i)
                                k2[i] = F1[i] - k1[i];
                            solve(k2);
                            for (int i = 0; i < m; ++i) {
                                k2[i] += k1[i];
                                y1[i] = y[i] + hh * k2[i];
                            }
                            Av[4] = eval_cont(s1, y1.data(), F2.data(),
                                              need_dyn_nodes ? dnv[4].data() : nullptr, true);
                            for (int i = 0; i < m; ++i)
                                k3[i] = F2[i] - e32 * (k2[i] - F1[i]) - 2.0 * (k1[i] - F0[i]) +
                                        hh * dR * Tt[i];
                            solve(k3);
                            for (int i = 0; i < m; ++i) {
                                const double e = hh / 6.0 * (k1[i] - 2.0 * k2[i] + k3[i]);
                                const double sc =
                                    ATOL_Y + RTOL_Y * std::max(std::fabs(y[i]), std::fabs(y1[i]));
                                err = std::max(err, std::fabs(e) / sc);
                            }
                            for (int q = 1; q <= 3; ++q) {
                                dense_y(NODE[q], ytmp.data());
                                Av[q] = eval_cont(t + NODE[q] * hh, ytmp.data(), nullptr,
                                                  need_dyn_nodes ? dnv[q].data() : nullptr, true);
                            }
                        }
                    } else {
                        for (int q = 1; q <= 3; ++q)
                            Av[q] = eval_cont(t + NODE[q] * hh, nullptr, nullptr,
                                              need_dyn_nodes ? dnv[q].data() : nullptr, true);
                        Av[4] = eval_cont(s1, nullptr, nullptr,
                                          need_dyn_nodes ? dnv[4].data() : nullptr, true);
                    }
                    if (err < 1e10) {
                        // The 5-point Lobatto rule (the quartic's own integral, exact to
                        // degree 7) against Simpson on its nodes 0, ½, 1: the difference
                        // is Simpson's error, and a sixteenth of it the error of
                        // Simpson on the halves, the level the quartic's partial
                        // integrals, which place the firings, are held to.
                        const double s_lob =
                            hh * ((Av[0] + Av[4]) / 20.0 + (Av[1] + Av[3]) * 49.0 / 180.0 +
                                  Av[2] * 16.0 / 45.0);
                        const double s_simp = hh / 6.0 * (Av[0] + 4.0 * Av[2] + Av[4]);
                        err = std::max(err, std::fabs(s_lob - s_simp) / 16.0 /
                                                (ATOL_H + RTOL_H * std::fabs(s_lob)));
                    }
                    if (!(err <= 1e10))
                        err = 1e10; // non-finite (an overflowing stage): shrink
                    // Local orders: 3 for the Rosenbrock y error, 5 for the quadrature's.
                    const double fac =
                        err > 0.0
                            ? std::clamp(0.9 * (m > 0 ? 1.0 / std::cbrt(err) : std::pow(err, -0.2)),
                                         0.2, 5.0)
                            : 5.0;
                    // A step at the floor that still fails is a jump in a rate or a
                    // rate rule's right-hand side (`x' = piecewise(-1, x > 0, 0)` at
                    // x = 0): no step resolves it, and one of hmin, a few ulps of t,
                    // crosses it with an error of that order. Accepted, a bounded
                    // number of times in a row; past that the right-hand side is
                    // singular, not discontinuous, and the run is refused.
                    // A step that leaves y exactly where it was while the rate rules
                    // say it moves has a stage on the far side of a jump in their
                    // right-hand side: `x' = piecewise(-1, x > 0, 0)` with x = 1e-9
                    // and the midpoint at x < 0, where the slope is 0. Its error
                    // estimate can still pass, at a step that then never moves x, so
                    // the run crawled forward a few ns a panel for ever. Shrink it
                    // until a stage stays on this side. At the floor no step can (x
                    // = 2e-14 at t = 3, where the floor is 4e-14): take that one by
                    // Euler, which crosses with an error of the floor times |f|.
                    if (m > 0 && err <= 1.0) {
                        // "Should have moved": the first stage alone, which carries
                        // the step's own damping of a stiff rule, moves y by several
                        // ulps. Near any steady state, stiff or not, hh·k1 is under
                        // an ulp of y and y not moving is right; at a jump the first
                        // stage is the slope on this side and the second cancels it.
                        // Per component: a clamp beside a rule that keeps moving
                        // (`z' = 1`) is stuck all the same.
                        bool stuck = false;
                        for (int i = 0; i < m; ++i) {
                            const double a = std::fabs(y[i]);
                            const double ulp =
                                std::nextafter(a, std::numeric_limits<double>::infinity()) - a;
                            stuck = stuck || (y1[i] == y[i] && std::fabs(hh * k1[i]) > 4.0 * ulp);
                        }
                        if (stuck && hh > hmin) {
                            h = std::max(0.25 * hh, hmin);
                            continue;
                        }
                        if (stuck) {
                            for (int i = 0; i < m; ++i) {
                                k1[i] = k2[i] = F0[i]; // dense y: the straight line
                                y1[i] = y[i] + hh * F0[i];
                            }
                            Av[4] = eval_cont(s1, y1.data(), F2.data(),
                                              need_dyn_nodes ? dnv[4].data() : nullptr, true);
                            for (int q = 1; q <= 3; ++q) {
                                dense_y(NODE[q], ytmp.data());
                                Av[q] = eval_cont(t + NODE[q] * hh, ytmp.data(), nullptr,
                                                  need_dyn_nodes ? dnv[q].data() : nullptr, true);
                            }
                        }
                    }
                    const bool at_floor = hh <= hmin && err < 1e10;
                    if (err <= 1.0 || (at_floor && floor_steps < MAX_FLOOR_STEPS)) {
                        floor_steps = err <= 1.0 ? 0 : floor_steps + 1;
                        h_next = std::min(hmax, to_stop ? std::max(h, hh * fac) : hh * fac);
                        break;
                    }
                    if (hh <= hmin) {
                        char buf[96];
                        std::snprintf(buf, sizeof buf, "t = %.17g: %d steps of %.3g in a row", t,
                                      MAX_FLOOR_STEPS, hmin);
                        throw std::runtime_error(
                            std::string(use_psa ? "PSA" : "SSA") +
                            ": the continuous part of the model (time-dependent rates or rate "
                            "rules) cannot be integrated past " +
                            buf +
                            " failed the error test. A rate or a rate rule is singular there.");
                    }
                    h = std::max(hh * fac, hmin);
                }
                // A run that makes no headway: panels at the floor (two hmin, a
                // hundred-odd ulps of t), a million in a row. A rate
                // or rate rule that is singular there does that; refuse it rather
                // than spin until a timeout, or for ever without one. (A fast
                // forcing over a long horizon takes short panels, not these.)
                if (hh <= 2.0 * hmin && hh < t_stop - t) {
                    if (++tiny_panels > MAX_TINY_PANELS) {
                        char buf[160];
                        std::snprintf(buf, sizeof buf,
                                      "%ld panels in a row shorter than %.3g at t = %.17g",
                                      MAX_TINY_PANELS, 2.0 * hmin, t);
                        throw std::runtime_error(
                            std::string(use_psa ? "PSA" : "SSA") +
                            ": the continuous part of the model makes no headway: " + buf +
                            ". A rate or a rate rule is singular there.");
                    }
                } else {
                    tiny_panels = 0;
                }
                s1p = (hh == t_stop - t) ? t_stop : t + hh;
                h_next_p = h_next;
                tp = t;
                th0 = 0.0;
                bank_th = 0.0;
                coeffs(Av, Ac);
            }
            carry = false;
            const double s1 = s1p;

            // Where in the panel does something discrete happen first?
            const double A_s = sel_total();
            if (!std::isfinite(A_s))
                refuse_nonfinite_propensity(-1, A_s, t);
            double th_fire = 2.0;
            const double L0 = L_at(th0);
            const double span = A_s * hh * (1.0 - th0) + (L_at(1.0) - L0);
            if (Hz + span >= E) {
                // The hazard reaches E inside the panel: solve the quintic by
                // Newton, kept inside a shrinking bracket (bisecting when a step
                // leaves it).
                auto G = [&](double th) {
                    return Hz + A_s * hh * (th - th0) + (L_at(th) - L0) - E;
                };
                auto q_at = [&](double th) {
                    double acc = 0.0;
                    for (int k = 4; k >= 0; --k)
                        acc = acc * th + Ac[k];
                    return acc;
                };
                double lo = th0, hi = 1.0;
                double th = th0 + (1.0 - th0) * (E - Hz) / span;
                if (!(th > lo && th < hi))
                    th = 0.5 * (lo + hi);
                th_fire = hi;
                for (int it = 0; it < 100; ++it) {
                    const double g = G(th);
                    if (std::fabs(g) <= 1e-15 * std::max(1.0, E)) {
                        th_fire = th; // converged on the root itself
                        break;
                    }
                    if (g > 0.0)
                        hi = th;
                    else
                        lo = th;
                    th_fire = hi;
                    if (hi - lo <= 1e-15)
                        break;
                    const double dg = hh * (A_s + q_at(th));
                    // The next Newton step is below θ's resolution here. The test
                    // on |g| above cannot see this: G's terms run to thousands
                    // (A_s·hh), so its rounding floor sits far above 1e-15, and a
                    // step that rounds onto the bracket's end would otherwise
                    // fall through to bisecting a bracket the root already sits
                    // at one end of.
                    if (dg > 0.0 && std::fabs(g) <= 1e-14 * dg) {
                        th_fire = th;
                        break;
                    }
                    double nt = dg > 0.0 ? th - g / dg : 0.5 * (lo + hi);
                    if (!(nt > lo && nt < hi))
                        nt = 0.5 * (lo + hi);
                    th = nt;
                }
            }
            double th_ev = 2.0;
            if (n_events > 0) {
                // Each trigger at the remaining quarter points of the panel, on the
                // dense y: the first interval in which any trigger leaves its
                // recorded truth is bisected (a rise fires, a fall re-arms).
                // Looking only at the panel's end missed a trigger true inside
                // it, and a panel can end on both edges of a time window.
                auto trigger_at = [&](double th, int ei) {
                    if (m > 0) {
                        dense_y(th, ytmp.data());
                        load_y(ytmp.data());
                    }
                    sync_state(th >= 1.0 ? s1 : tp + th * hh);
                    return eval_ref.evaluate(events[ei].trigger_expr_idx) > 0.5;
                };
                probe_prev.assign(trigger_was_true.begin(), trigger_was_true.end());
                probe_cur.assign(static_cast<std::size_t>(n_events), 0);
                double th_prev = th0;
                // With rate rules any trigger may read a target that moves.
                std::vector<int> all_triggers;
                if (m > 0)
                    for (int ei = 0; ei < n_events; ++ei)
                        all_triggers.push_back(ei);
                const std::vector<int> &watched = m > 0 ? all_triggers : time_triggers;
                for (int q = 1; q <= 4 && th_ev > 1.0; ++q) {
                    const double thq = q == 4 ? 1.0 : th0 + (1.0 - th0) * 0.25 * q;
                    if (m > 0) {
                        dense_y(thq, ytmp.data());
                        load_y(ytmp.data());
                    }
                    sync_state(q == 4 ? s1 : tp + thq * hh);
                    for (int ei : watched)
                        probe_cur[ei] = eval_ref.evaluate(events[ei].trigger_expr_idx) > 0.5;
                    for (int ei : watched) {
                        if (probe_cur[ei] == probe_prev[ei])
                            continue;
                        const bool at_lo = probe_prev[ei] != 0;
                        double lo = th_prev, hi = thq;
                        for (int it = 0; it < 200; ++it) {
                            const double mid = 0.5 * (lo + hi);
                            const double tmid = tp + mid * hh;
                            if (mid <= lo || mid >= hi || hh * (hi - lo) <= bisect_tol(tmid))
                                break;
                            if (trigger_at(mid, ei) != at_lo)
                                hi = mid;
                            else
                                lo = mid;
                        }
                        th_ev = std::min(th_ev, hi);
                    }
                    th_prev = thq;
                }
            }
            const bool event_first = th_ev <= 1.0 && th_ev <= th_fire;
            const bool fire_first = !event_first && th_fire <= 1.0;
            const double th_cut = event_first ? th_ev : (fire_first ? th_fire : 1.0);
            const double s_cut = th_cut >= 1.0 ? s1 : tp + th_cut * hh;

            // Rows strictly before the change (a row within bisection precision
            // of an event records the post-event state, as the old loop did).
            while (next_output < n_out) {
                const double tk = t_out[next_output];
                const bool before = event_first ? tk < s_cut - sample_event_tol(s_cut)
                                                : (fire_first ? tk < s_cut : tk <= s1);
                if (!before)
                    break;
                record_at(tk, (tk - tp) / hh);
            }
            bank_dyn(th_cut);
            Hz += A_s * (s_cut - t) + (L_at(th_cut) - L0);

            if (!event_first && !fire_first) {
                // Nothing discrete in the panel: move to its end.
                t = s1;
                y = y1;
                A0 = Av[4];
                F0 = F2;
                if (need_dyn_nodes)
                    dnv[0] = dnv[4];
                if (m > 0)
                    load_y(y.data());
                jac_ok = false;
                h = h_next_p;
                psa_now = t;
                model.set_current_time(t);
                if (s1 == t_stop && t_stop < times.t_end)
                    fresh = true; // a breakpoint: a rate may jump here
                continue;
            }

            // Move to the change.
            if (m > 0) {
                dense_y(th_cut, ytmp.data());
                y = ytmp;
                load_y(y.data());
            }
            t = s_cut;
            psa_now = t;
            model.set_current_time(t);
            sync_state(t);
            h = std::max(h_next_p, 64.0 * hmin);
            fresh = true;

            if (event_first) {
                obs_current = true; // sync_state above
                if (fire_rising_edges(t)) {
                    recompute_all_propensities();
                    for (int i = 0; i < m; ++i) // an event may assign a rate-rule target
                        y[i] = conc[rate_rule_odes[i].target0];
                }
                continue;
            }

            // A firing at t: which reaction, from the propensities at t.
            Hz = 0.0;
            E = draw_exp();
            double a_dyn = 0.0;
            for (int k = 0; k < nd; ++k) {
                const int r = dyn[k];
                const double v = model.compute_propensity(r, conc.data());
                if (!std::isfinite(v))
                    refuse_nonfinite_propensity(r, v, t);
                rxn_dir[r] = v < 0.0 ? -1 : 1;
                dnow[k] = (v < 0.0 ? -v : v) * (use_psa ? scaling_factors[r] : 1.0);
                a_dyn += dnow[k];
            }
            const double A_now = sel_total();
            const double a_all = A_now + a_dyn;
            if (!(a_all > 0.0))
                continue; // the rates vanished at the crossing: nothing can fire
            const double u = next_u01() * a_all;
            int selected = -1;
            if (u < A_now) {
                selected = sel_find(u);
            } else {
                double cum = A_now;
                for (int k = 0; k < nd; ++k) {
                    if (dnow[k] <= 0.0)
                        continue;
                    selected = dyn[k];
                    cum += dnow[k];
                    if (cum > u)
                        break;
                }
            }
            if (selected < 0)
                continue;
            apply_firing(selected);
            ++total_steps;
            obs_current = false;
            if (n_events == 0 && th_cut < 1.0 && fire_keeps_panel[selected]) {
                // Nothing a dynamic propensity reads moved, and no trigger is
                // watching: stay in this panel. The static propensities the
                // firing moved are all that need refreshing; when none of them
                // reads a function, the observables and functions wait for the
                // next sync.
                if (lean_ok[selected]) {
                    refresh_jit_propensities();
                    for (int r : dep_graph.affected_reactions(selected))
                        set_propensity(r);
                } else {
                    refresh_after_firing(selected);
                }
                carry = true;
                fresh = false;
                th0 = th_cut;
                synced = false;
                continue;
            }
            refresh_after_firing(selected);
            obs_current = dep_graph.has_functional_rates; // refresh updated them
            const bool drained = n_events > 0 && fire_rising_edges(t);
            if (drained) {
                recompute_all_propensities();
                for (int i = 0; i < m; ++i)
                    y[i] = conc[rate_rule_odes[i].target0];
            }
            if (!drained && th_cut < 1.0 && fire_keeps_panel[selected]) {
                // Nothing a_d reads moved: stay in this panel.
                carry = true;
                fresh = false;
                th0 = th_cut;
                synced = false;
                continue;
            }
            // Every dynamic propensity is Functional or reads a rate-rule target,
            // and a rate rule is itself a Functional reaction, so the refresh
            // above (and any event drain) left the observables and functions at
            // this state: the next panel's start needs no second evaluation.
            synced = dep_graph.has_functional_rates;
        }
        // Rows at t_end not yet written (a row at exactly t_end whose panel
        // ended in an event).
        hh = 0.0; // the state is at t: no dense step, no partial integral
        bank_th = 0.0;
        while (next_output < n_out)
            record_at(t_out[next_output], 0.0);
        if (m > 0)
            load_y(y.data());
    } else
        // The discrete loop: no rate reads time and there is no rate rule, so every
        // propensity is constant between firings and the draw is exact as it stands.
        while (next_output < n_out) {
            if (budget.active() && ++steps_since_timeout_check >= TIMEOUT_CHECK_STRIDE) {
                budget.check();
                steps_since_timeout_check = 0;
            }
            if (pc_due(t))
                pc_refresh_at(t);

            // 1. Total propensity: the sum tree's root, O(1), or an O(N) flat sum.
            double a0 = sel_total();
            if (!std::isfinite(a0)) // issue #809; each propensity was finite when it was set
                refuse_nonfinite_propensity(-1, a0, t);

            // 2. If total propensity is zero, the reaction system is stuck —
            //    but a time-only event trigger could still fire. Probe within
            //    (t, t_end]; if a crossing exists, advance to it and fire.
            //    Otherwise, fast-forward all remaining samples at the current
            //    state.
            if (a0 <= 0.0) {
                // A piecewise-constant rate may turn on at the next breakpoint:
                // look no further than that.
                const double t_lim = pc_refresh ? std::min(times.t_end, pc_next()) : times.t_end;
                if (n_events > 0) {
                    // Only the time: the batch is taken from every trigger below.
                    const double t_event_idle = probe_events_in_window(t, t_lim);
                    if (std::isfinite(t_event_idle) && t_event_idle <= t_lim) {
                        // Record any samples strictly before t_event_idle.
                        while (next_output < n_out && t_event_idle > t_out[next_output] &&
                               t_out[next_output] < t_event_idle - sample_event_tol(t_event_idle)) {
                            const double *cc = conc.data();
                            model.update_observables(cc);
                            for (int j = 0; j < n_obs; ++j) {
                                obs_buf[j] = model.observables()[j].total;
                            }
                            result.record(next_output, t_out[next_output], cc, obs_buf.data());
                            if (rec_stats)
                                rs_record(next_output, t_out[next_output]);
                            if (n_func > 0) {
                                model.evaluate_functions(t_out[next_output]);
                                auto fvals = model.function_values();
                                result.record_expressions(next_output, fvals.data());
                                pc_guard(t_out[next_output]);
                            }
                            ++next_output;
                        }
                        if (next_output >= n_out)
                            break;
                        t = t_event_idle;
                        psa_now = t; // GH #15 — keep the PSA dwell clock on t
                        sync_state(t);
                        const bool fired = fire_rising_edges(t);
                        if (fired) {
                            recompute_all_propensities();
                        }
                        model.set_current_time(t);
                        continue;
                    }
                }
                if (t_lim < times.t_end) {
                    // Nothing fires before the breakpoint: rows up to it hold this
                    // state, and the rates are re-read there (at the loop's top).
                    while (next_output < n_out && t_out[next_output] <= t_lim) {
                        model.update_observables(conc.data());
                        for (int j = 0; j < n_obs; ++j)
                            obs_buf[j] = model.observables()[j].total;
                        result.record(next_output, t_out[next_output], conc.data(), obs_buf.data());
                        if (rec_stats)
                            rs_record(next_output, t_out[next_output]);
                        model.evaluate_functions(t_out[next_output]);
                        auto fvals = model.function_values();
                        result.record_expressions(next_output, fvals.data());
                        pc_guard(t_out[next_output]);
                        ++next_output;
                    }
                    if (next_output >= n_out)
                        break;
                    t = t_lim;
                    psa_now = t;
                    model.set_current_time(t);
                    continue;
                }
                // Truly stuck — fast-forward.
                model.update_observables(conc.data());
                for (int j = 0; j < n_obs; ++j) {
                    obs_buf[j] = model.observables()[j].total;
                }
                while (next_output < n_out) {
                    result.record(next_output, t_out[next_output], conc.data(), obs_buf.data());
                    if (rec_stats)
                        rs_record(next_output, t_out[next_output]);
                    // Issue #569 — same pairing as every other recording site above.
                    // Evaluated per sample rather than once before the loop, because
                    // arriving here does NOT mean the functions are constant: the
                    // fast-forward is entered on a0 == 0 with no live reaction left,
                    // which says nothing about a function that reads time() — an
                    // output-only function, or one whose reaction is exhausted. Its
                    // recorded column must keep tracking t across the frozen tail; a
                    // single pre-loop evaluation would flatline it at one value.
                    if (n_func > 0) {
                        model.evaluate_functions(t_out[next_output]);
                        auto fvals = model.function_values();
                        result.record_expressions(next_output, fvals.data());
                        pc_guard(t_out[next_output]);
                    }
                    ++next_output;
                }
                break;
            }

            // 3. Sample time to next reaction: tau = -ln(r1) / a0
            double r1 = next_u01();
            while (r1 == 0.0)
                r1 = next_u01(); // avoid log(0)
            double tau = -std::log(r1) / a0;
            double t_proposed = t + tau;
            // A breakpoint first: a piecewise-constant rate jumps there, so stop,
            // re-read the rates and redraw (exact by memorylessness).
            const bool pc_cap = pc_refresh && t_proposed > pc_next();
            if (pc_cap)
                t_proposed = pc_next();

            // 4. Detect event-trigger crossings within (t, t_proposed].
            //    State is piecewise-constant during τ, so a time-dependent trigger
            //    is a 1-D function of t and is well-resolved by bisection.
            //    State-dependent triggers (no time component) cannot flip during
            //    τ — those are handled post-fire below.
            // Only the time: the batch is taken from every trigger below.
            const double t_event = probe_events_in_window(t, t_proposed);

            // At the breakpoint itself an event still fires: the stop there is
            // not a firing, and the event's rows then read its effect, as
            // without the stop.
            const bool event_wins = std::isfinite(t_event) &&
                                    (t_event < t_proposed || (pc_cap && t_event <= t_proposed));
            double t_advance = event_wins ? t_event : t_proposed;

            // 5. Record output points strictly before t_advance. When an event
            //    lands at (or within bisection precision of) a sample time,
            //    defer that sample so it records post-event state on the next
            //    iteration — matches ODE rootfind semantics. The tolerance is
            //    several orders wider than bisect_tol so a legitimate sample
            //    strictly before t_event is still recorded pre-event.
            while (next_output < n_out && t_advance >= t_out[next_output]) {
                if (event_wins && t_out[next_output] >= t_event - sample_event_tol(t_event))
                    break;
                const double *cc = conc.data();
                model.update_observables(cc);
                for (int j = 0; j < n_obs; ++j) {
                    obs_buf[j] = model.observables()[j].total;
                }
                result.record(next_output, t_out[next_output], cc, obs_buf.data());
                if (rec_stats)
                    rs_record(next_output, t_out[next_output]);
                if (n_func > 0) {
                    model.evaluate_functions(t_out[next_output]);
                    auto fvals = model.function_values();
                    result.record_expressions(next_output, fvals.data());
                    pc_guard(t_out[next_output]);
                }
                ++next_output;
            }

            if (next_output >= n_out)
                break;

            if (pc_cap && !event_wins) {
                // The breakpoint came first: nothing fired. The rates are re-read
                // at the loop's top.
                t = t_proposed;
                psa_now = t;
                model.set_current_time(t);
                continue;
            }

            if (event_wins) {
                // Advance time to t_event, fire the batch, redraw τ. The
                // candidate reaction is NOT fired — its tentative τ assumed the
                // pre-event state and is discarded on the redraw.
                t = t_event;
                psa_now = t; // GH #15 — keep the PSA dwell clock on t
                sync_state(t);
                const bool fired = fire_rising_edges(t);
                if (fired) {
                    recompute_all_propensities();
                }
                model.set_current_time(t);
                continue;
            }

            // 6. Select reaction: O(log N) sum-tree descent, or O(N) flat scan.
            double r2 = next_u01() * a0;
            int selected = sel_find(r2);
            if (selected < 0)
                selected = 0;
            if (selected >= nr)
                selected = nr - 1;

            apply_firing(selected);

            // 8. Advance time unconditionally so time() stays current
            t = t_proposed;
            psa_now = t; // GH #15 — keep the PSA dwell clock on t
            model.set_current_time(t);
            ++total_steps;

            refresh_after_firing(selected);

            // 10. State-dependent triggers can flip false→true after this fire.
            //     (Time-only triggers were already detected by bisection above.)
            //     Sweep all events; fire rising edges through process_firing_batch.
            if (n_events > 0 && fire_rising_edges(t))
                recompute_all_propensities();
        }

    // ─── Write final state back to model ─────────────────────────────────────
    //
    // The discrete loop ends at the last firing it took, somewhere before t_end;
    // nothing fires in (t, t_end], so every count is already its t_end value. The
    // continuous loop runs to t_end itself, rate-rule targets and the events
    // they trigger included (issues #718, #751).
    if (t < times.t_end)
        t = times.t_end;
    {
        auto &species = const_cast<std::vector<Species> &>(model.species());
        for (int i = 0; i < ns; ++i) {
            species[i].concentration = conc[i];
        }
        model.set_current_time(t);
        // ...and the event state that goes with it (issue #693). No execution
        // is ever pending: a delayed event is refused above.
        NetworkModel::EventCarry carry;
        carry.valid = true;
        carry.t = t;
        carry.trigger.assign(trigger_was_true.begin(), trigger_was_true.end());
        const bool continued = model.event_carry_for(times.t_start, n_events) != nullptr;
        model.publish_event_carry(std::move(carry), continued);
    }

    // Solver stats
    result.solver_stats().n_steps = static_cast<int>(total_steps);
    result.solver_stats().n_rhs_evals = static_cast<int>(total_steps);

    // GH #110 — boundary diagnostics. Resolve the first-offender indices to
    // human-readable labels here, out of the hot loop. The Python layer turns
    // nonzero counts into one filterable warning each.
    {
        auto &diag = result.ssa_diagnostics();
        diag.propensity_backend = prop_backend; // GH #190 — accurate backend reporting
        diag.n_negative_crossings = neg_cross_count;
        if (first_neg_species >= 0) {
            const auto &names = model.species_names();
            if (first_neg_species < static_cast<int>(names.size()))
                diag.first_negative_species = names[first_neg_species];
        }
        diag.n_reverse_fires = reverse_fire_count;
        if (first_reverse_rxn >= 0)
            diag.first_reverse_reaction = reaction_label(model, reactions[first_reverse_rxn]);
        diag.n_rounded_populations = n_rounded; // issue #718
        if (first_rounded >= 0)                 // the one name, not a copy of every species' name
            diag.first_rounded_species = model.species()[first_rounded].name;
    }

    // GH #15 — PSA partial-scaling diagnostics. Close every reaction's dwell over
    // the final [psa_last_t[r], t] tail, then publish the run integrals. Left
    // untouched (defaults) on exact SSA so non-PSA results are unchanged.
    if (use_psa) {
        psa_now = t;
        for (int r = 0; r < nr; ++r)
            psa_flush(r);

        auto &diag = result.ssa_diagnostics();
        diag.psa_active = true;
        diag.psa_time = t - times.t_start;
        diag.psa_exact_event_integral = psa_exact_int;
        diag.psa_scaled_event_integral = psa_scaled_int;
        diag.psa_peak_population = psa_peak_pop;
        diag.psa_activation_crossed = (psa_peak_pop >= 2.0 * poplevel);
        diag.psa_reaction_index.resize(nr);
        diag.psa_mbar_integral.resize(nr);
        diag.psa_qexc_integral.resize(nr);
        for (int r = 0; r < nr; ++r) {
            diag.psa_reaction_index[r] = reactions[r].index;
            diag.psa_mbar_integral[r] = psa_mbar_int[r];
            diag.psa_qexc_integral[r] = psa_qexc_int[r];
        }
    }

    return result;
}

} // namespace bngsim
