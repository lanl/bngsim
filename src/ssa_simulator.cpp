// bngsim/src/ssa_simulator.cpp — SSA and PSA stochastic simulators
//
// SSA: Gillespie's direct method with dependency graph + Fenwick tree.
// PSA: Partial Scaling Algorithm (Lin, Feng, Hlavacek, J. Chem. Phys. 150, 244101, 2019).
//
// Optimizations:
//   1. Dependency graph: After reaction fires, only recompute propensities
//      for reactions whose propensity is affected by the changed species.
//      O(k) where k ≈ 5–20, instead of O(N) over all reactions.
//   2. Fenwick tree (binary indexed tree): O(log N) reaction selection
//      and O(log N) propensity updates, replacing O(N) linear scan.
//
// Per-instance RNG (std::mt19937_64). Deterministic seeding.
// Simulation time is always tracked so time()/t() expressions stay current.

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
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

namespace bngsim {

// ─── Fenwick Tree (Binary Indexed Tree) ──────────────────────────────────────
//
// Supports O(log N) operations for:
//   - point_update(i, delta): add delta to element i
//   - prefix_sum(i): sum of elements 0..i (inclusive)
//   - total(): sum of all elements
//   - find(target): smallest index where prefix_sum >= target (O(log N))
//
// Used for reaction selection: stores propensities (or scaled propensities
// for PSA), enabling O(log N) sampling instead of O(N) linear scan.

class FenwickTree {
  public:
    FenwickTree() : n_(0) {}

    explicit FenwickTree(int n) : n_(n), tree_(n + 1, 0.0), vals_(n, 0.0) {}

    // Set element i to value (compute delta internally)
    void set(int i, double value) {
        double delta = value - vals_[i];
        vals_[i] = value;
        point_update(i, delta);
    }

    // Add delta to element i: O(log N)
    void point_update(int i, double delta) {
        for (int j = i + 1; j <= n_; j += j & (-j)) {
            tree_[j] += delta;
        }
    }

    // Sum of elements 0..i (inclusive): O(log N)
    double prefix_sum(int i) const {
        double s = 0.0;
        for (int j = i + 1; j > 0; j -= j & (-j)) {
            s += tree_[j];
        }
        return s;
    }

    // Total sum of all elements: O(log N)
    double total() const { return prefix_sum(n_ - 1); }

    // Find smallest index where prefix_sum >= target: O(log N)
    // Uses binary lifting (bit-by-bit descent), not binary search.
    // This is the standard Fenwick tree "find" operation.
    int find(double target) const {
        int pos = 0;
        double sum = 0.0;
        // Find highest bit
        int bit = 1;
        while (bit <= n_)
            bit <<= 1;
        bit >>= 1;

        while (bit > 0) {
            int next = pos + bit;
            if (next <= n_ && sum + tree_[next] < target) {
                pos = next;
                sum += tree_[next];
            }
            bit >>= 1;
        }
        return pos; // 0-based index
    }

    // Get raw value at index i
    double value(int i) const { return vals_[i]; }

    int size() const { return n_; }

  private:
    int n_;
    std::vector<double> tree_; // 1-indexed Fenwick tree
    std::vector<double> vals_; // 0-indexed raw values (for computing deltas)
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

    // Build dependency graph from model
    void build(const NetworkModel &model) {
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
    // The Fenwick tree holds per-state propensities and is still built per-run
    // (its O(nr) allocation is trivial next to the graph build).
    DependencyGraph dep_graph;
    bool dep_graph_built = false;
    int dep_graph_ns = -1; // topology fingerprint guarding stale reuse
    int dep_graph_nr = -1;

    // GH #190 — path to the cc-compiled value-specialized propensity-vector .so
    // (symbol bngsim_ssa_propensities), produced by the Python codegen layer
    // (_codegen.prepare_ssa_propensity_lib) and content-cached on disk. When set
    // and the model is recompute-all eligible (pure mass-action exact SSA, no
    // events, small nr), run_internal loads it and takes the RR-style
    // recompute-all + flat-scan loop by DEFAULT — no MIR required. Empty ⇒ the
    // model wasn't eligible / codegen was skipped, and the incremental Fenwick
    // path runs unchanged.
    std::string propensity_lib_path;

    Impl(NetworkModel &m) : model(m) {}

    // Build the dependency graph if not already cached for the current topology.
    // The (ns, nr) fingerprint defends against a model whose structure changed
    // under the simulator (unusual — topology is normally fixed once loaded).
    DependencyGraph &dependency_graph() {
        const int ns = model.n_species();
        const int nr = model.n_reactions();
        if (!dep_graph_built || dep_graph_ns != ns || dep_graph_nr != nr) {
            dep_graph.build(model);
            dep_graph_built = true;
            dep_graph_ns = ns;
            dep_graph_nr = nr;
        }
        return dep_graph;
    }
};

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
//     N_min^r = min population among REACTANT species only (Eq. 14)
//     λ_r = 1 / max(1, ⌊N_min^r / N_c⌋)
//     scaled_rate_r = λ_r * propensity_r
//   When reaction r fires:
//     Update each species s by (1/λ_r) * ξ_{r,s}
//
// Optimizations:
//   - Dependency graph: only recompute propensities for affected reactions
//   - Fenwick tree: O(log N) reaction selection + O(log N) propensity updates

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

    // Reject delay-bearing events. Phase 5b lands EventNoDelay only; Phase 5c
    // tracks delay support (see dev/plans/SBML_SSA_SUPPORT_PLAN.md §5
    // Phase 5c). Surface a clear error here rather than at Simulator
    // construction so the message includes the event id.
    {
        const auto &evs = model.events();
        for (const auto &ev : evs) {
            if (ev.delay > 0.0 || ev.delay_expr_idx >= 0) {
                throw std::runtime_error("Event '" + ev.id +
                                         "' has a delay, which is not yet supported under SSA/PSA. "
                                         "Phase 5c of the SBML SSA Support Plan tracks delay "
                                         "support; for now, use method='ode' or remove the "
                                         "delay attribute.");
            }
        }
    }

    // Per-instance RNG with deterministic seed
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

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

    // GH #190 — reaction-selection structure (opt-in; default Fenwick). bngsim
    // selects with a Fenwick tree: O(log n) total/find plus an O(log n) update
    // per affected reaction. For SMALL reaction counts a flat cumulative array
    // (O(n) total + O(n) linear scan, but a single contiguous, branch-predictable,
    // L1-resident pass — RoadRunner's direct method) wins the per-selection
    // constant factors: the microbench (dev/notes/gh190_select_microbench.cpp)
    // measured 1.4-1.9x on the isolated selection workload for nr<=44. BUT once
    // the per-step affected-set sort/dedup is precomputed (below), selection is a
    // small fraction of per-step cost and the flat win washes out end-to-end
    // (flat≈Fenwick within noise on the high-activity suite models). So the flat
    // path stays OPT-IN — validated and bit-identical, useful for selection-bound
    // regimes and ablation, but not the default (no measured end-to-end gain, and
    // the index-order sum is a different—if equivalent—realization). Select with:
    //   unset/"fenwick"   Fenwick tree (default)
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
        return false; // "fenwick" / unrecognized → Fenwick tree (default)
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

    // Working arrays: species populations (0-based)
    std::vector<double> conc(ns);
    for (int i = 0; i < ns; ++i) {
        const auto &sp = model.species()[i];
        conc[i] = round_initial_population_to_storage(sp.concentration, sp.volume_factor);
    }

    // Propensity arrays
    std::vector<double> propensities(nr, 0.0);    // unscaled propensities (magnitudes)
    std::vector<double> scaling_factors(nr, 1.0); // λ_r per reaction (PSA)

    // GH #110 — sign-split firing direction per reaction. +1 forward (rate >= 0,
    // reactants → products), -1 reverse (rate < 0, products → reactants). The
    // Fenwick tree and propensities[] hold |rate| for selection; rxn_dir[r]
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

    // ─── Build (or reuse) dependency graph ───────────────────────────────────
    // Cached on the Impl across run() calls — topology-only, seed-independent.
    DependencyGraph &dep_graph = impl_->dependency_graph();

    // ─── Build Fenwick tree ──────────────────────────────────────────────────
    // Stores effective propensities: unscaled for SSA, scaled for PSA.
    FenwickTree ftree(nr);

    // GH #190 — flat cumulative-propensity array (mirrors what goes into the
    // Fenwick tree). Allocated only on the flat selection path; sel_set/sel_total/
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
    // Smallest index where the cumulative propensity reaches `target`. The flat
    // path scans left-to-right in index order (the same selection semantics as
    // the tree's binary lifting), terminating early once the running sum passes
    // the target; for small nr this contiguous pass beats the tree's descent.
    auto sel_find = [&](double target) -> int {
        if (!use_flat_select)
            return ftree.find(target);
        double cum = 0.0;
        for (int r = 0; r < nr; ++r) {
            cum += flat_eff[r];
            if (cum >= target)
                return r;
        }
        return nr - 1;
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
    // Delay support is deferred to Phase 5c: validate_for_ssa() (or the
    // Simulator init-time check) raises on any event with delay > 0 or
    // delay_expr_idx >= 0 before we ever reach this loop.
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

    // Helper: (re)compute one reaction's propensity, direction, PSA scaling,
    // and Fenwick-tree entry from the current conc[]. GH #110: the rate law is
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
        // (forward Euler) below. Keep it out of the Fenwick selection so it never
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
        int dir = (signed_prop < 0.0) ? -1 : 1;
        double prop = (dir < 0) ? -signed_prop : signed_prop; // |signed_prop|
        rxn_dir[r] = dir;
        propensities[r] = prop;

        double effective_prop = prop;
        if (use_psa && prop > 0.0) {
            const auto &rxn = reactions[r];
            const auto &consumed = (dir > 0) ? rxn.reactant_indices : rxn.product_indices;
            double n_min = std::numeric_limits<double>::max();
            for (int ci : consumed) {
                int si = ci - 1;
                if (si >= 0 && si < ns) {
                    n_min = std::min(n_min, conc[si] * species_list[si].volume_factor);
                }
            }
            if (n_min == std::numeric_limits<double>::max())
                n_min = 0.0;
            double floor_ratio = std::floor(n_min / poplevel);
            double inv_lambda = std::max(1.0, floor_ratio);
            scaling_factors[r] = 1.0 / inv_lambda;
            effective_prop = scaling_factors[r] * prop;
        } else if (use_psa) {
            scaling_factors[r] = 1.0;
            effective_prop = 0.0;
        }
        sel_set(r, effective_prop);
    };

    // Helper: recompute ALL propensities + Fenwick-tree entries. Called after
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

    // dX/dt for every rate-rule target, snapshotted at the current (already
    // synced) state. compute_propensity returns the rule RHS f directly — these
    // reactions carry ssa_volume_factor=1 and no reactants, so the "propensity"
    // is exactly f — and the per-species branch reproduces compute_derivs'
    // `f / V_c(X)` (hOSU rate-rule targets store amount/V_c).
    std::vector<double> rr_deriv(rate_rule_odes.size(), 0.0);
    auto snapshot_rr = [&]() {
        for (std::size_t i = 0; i < rate_rule_odes.size(); ++i) {
            const auto &rr = rate_rule_odes[i];
            double f = model.compute_propensity(rr.rxn, conc.data());
            rr_deriv[i] = rr.per_species ? f / rr.vf : f;
        }
    };
    // Advance every rate-rule target by forward Euler over `dt`, holding dX/dt at
    // the last snapshot (piecewise-constant across the sub-step, exactly as the
    // propensities are). Writing into conc[] makes the target appear in the
    // recorded trajectory and lets the next sync_state see the advanced value.
    auto integrate_rr = [&](double dt) {
        if (dt <= 0.0)
            return;
        for (std::size_t i = 0; i < rate_rule_odes.size(); ++i)
            conc[rate_rule_odes[i].target0] += rr_deriv[i] * dt;
    };

    // process_firing_batch — adapted from cvode_simulator.cpp:1067-1186.
    // SBML L3 simultaneous-event semantics: priority-ordered drain with
    // state refresh between immediate fires; cancellation of non-persistent
    // entries whose trigger reverts post-fire; UVFTT pre-snapshot of
    // assignment RHS values. Returns true if any event modified conc[].
    auto process_firing_batch = [&](double t_now, const std::vector<int> &firing_in) -> bool {
        if (firing_in.empty())
            return false;

        std::vector<std::vector<double>> snapshot_vals(firing_in.size());
        for (size_t k = 0; k < firing_in.size(); ++k) {
            const auto &ev = events[firing_in[k]];
            if (ev.use_values_from_trigger_time) {
                snapshot_vals[k].reserve(ev.assignments.size());
                for (const auto &[sp_idx0, val_expr_idx] : ev.assignments) {
                    (void)sp_idx0;
                    snapshot_vals[k].push_back(eval_ref.evaluate(val_expr_idx));
                }
            }
            // Delays are gated out at Simulator init; ignore delay fields here.
        }

        std::vector<bool> done(firing_in.size(), false);
        bool any_fired = false;

        auto eval_pri = [&](size_t k) -> double {
            const auto &ev = events[firing_in[k]];
            return (ev.priority_expr_idx >= 0) ? eval_ref.evaluate(ev.priority_expr_idx)
                                               : static_cast<double>(ev.priority);
        };

        while (true) {
            ssize_t best = -1;
            double best_pri = 0.0;
            for (size_t k = 0; k < firing_in.size(); ++k) {
                if (done[k])
                    continue;
                double pk = eval_pri(k);
                if (best < 0 || pk > best_pri) {
                    best = static_cast<ssize_t>(k);
                    best_pri = pk;
                }
            }
            if (best < 0)
                break;
            size_t k = static_cast<size_t>(best);
            done[k] = true;

            int ei = firing_in[k];
            const auto &ev = events[ei];
            const auto &assigns = ev.assignments;
            std::vector<double> nv(assigns.size());
            if (ev.use_values_from_trigger_time) {
                nv = snapshot_vals[k];
            } else {
                for (size_t a = 0; a < assigns.size(); ++a) {
                    nv[a] = eval_ref.evaluate(assigns[a].second);
                }
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
                    conc[sp_idx0] = nv[a];
                    sp_vec_ref[sp_idx0].concentration = nv[a];
                }
            }
            any_fired = true;

            model.update_observables(conc.data());
            model.evaluate_functions(t_now);

            for (size_t k2 = 0; k2 < firing_in.size(); ++k2) {
                if (done[k2])
                    continue;
                const auto &ev_k = events[firing_in[k2]];
                if (ev_k.persistent)
                    continue;
                double tv = eval_ref.evaluate(ev_k.trigger_expr_idx);
                if (tv <= 0.5)
                    done[k2] = true;
            }
        }
        return any_fired;
    };

    // Helper: probe a trigger expression at a given time. Bisects within
    // (lo, hi] to find the rising-edge t_cross to BISECT_EPS precision. Pre:
    // trigger is FALSE at lo and TRUE at hi (with current conc[]). Returns
    // hi (the smallest known-true point).
    constexpr double BISECT_EPS = 1e-12;
    // Sample-vs-event-time tolerance. Several orders of magnitude wider
    // than BISECT_EPS so the recording loop defers a sample whose time is
    // within bisection imprecision of an event crossing, but not legitimate
    // samples strictly before the event. Defined at function scope so both
    // the a0==0 idle path and the τ-step recording loop can reuse it.
    constexpr double SAMPLE_EVENT_TOL = 1e-9;
    auto bisect_trigger = [&](int ei, double lo, double hi) -> double {
        // State is unchanged during the τ-step; only time advances.
        while (hi - lo > BISECT_EPS) {
            double mid = 0.5 * (lo + hi);
            model.evaluate_functions(mid);
            double v = eval_ref.evaluate(events[ei].trigger_expr_idx);
            if (v > 0.5) {
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
    if (n_events > 0) {
        sync_state(times.t_start);

        std::vector<int> t0_firing;
        t0_firing.reserve(n_events);
        for (int i = 0; i < n_events; ++i) {
            trigger_was_true[i] = events[i].initial_value;
            double val = eval_ref.evaluate(events[i].trigger_expr_idx);
            bool now_true = (val > 0.5);
            if (now_true && !trigger_was_true[i]) {
                t0_firing.push_back(i);
            }
            trigger_was_true[i] = now_true;
        }

        if (process_firing_batch(times.t_start, t0_firing)) {
            // Re-sync trigger_was_true against post-fire state so any event
            // that falsified its own trigger can re-arm on the next rise.
            for (int ei = 0; ei < n_events; ++ei) {
                double v = eval_ref.evaluate(events[ei].trigger_expr_idx);
                trigger_was_true[ei] = (v > 0.5);
            }
        }
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
    // Detect this by checking whether any function value moves when only time
    // advances (concentrations held fixed across the probes). If so, switch the
    // main loop to piecewise-constant sub-stepping: cap each step at dt_max and
    // re-evaluate all functions + propensities at the new time, the same way the
    // ODE RHS refreshes assignment rules on every call. For piecewise-constant
    // propensities the discard-and-resample at the cap is exact (exponential
    // memorylessness); the only approximation is holding the rate constant over
    // dt_max, which vanishes as dt_max → 0.
    bool time_dependent_rates = false;
    if (model.n_functions() > 0) {
        const double span = times.t_end - times.t_start;
        const double h = (span > 0.0) ? span : 1.0;
        const double probe_times[3] = {times.t_start, times.t_start + 0.5 * h, times.t_start + h};
        std::vector<double> base;
        for (int pi = 0; pi < 3 && !time_dependent_rates; ++pi) {
            model.evaluate_functions(probe_times[pi]);
            auto vals = model.function_values();
            if (pi == 0) {
                base = std::move(vals);
            } else {
                for (size_t fi = 0; fi < vals.size() && fi < base.size(); ++fi) {
                    if (std::fabs(vals[fi] - base[fi]) > 1e-12 * (1.0 + std::fabs(base[fi]))) {
                        time_dependent_rates = true;
                        break;
                    }
                }
            }
        }
        // Restore function-driven parameters to their t_start values; the
        // initial-state record below re-evaluates at t_start regardless.
        model.evaluate_functions(times.t_start);
    }
    // GH #81 — a rate-rule ODE makes every propensity that reads its target a
    // continuously-varying function of time (the target moves between fires),
    // so the piecewise-constant sub-stepping MUST engage even when the time-only
    // probe above found no moving function value (a rate rule whose RHS reads
    // only species has none). This is also where the deterministic Euler
    // integration of the targets is driven (one Euler step per sub-step).
    if (has_rate_rules)
        time_dependent_rates = true;
    // Sub-step cap for time-dependent rates: resolve the rate variation over
    // the simulation horizon. 1000 sub-steps tracks the smooth exponential
    // assignment rules in the RC#2 corpus within stochastic tolerance, and is
    // the forward-Euler step for rate-rule targets (GH #81).
    const double time_dep_dt_max =
        time_dependent_rates ? (times.t_end - times.t_start) / 1000.0 : 0.0;

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
            !use_psa && n_events == 0 && !time_dependent_rates && !has_rate_rules;
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

    // Species buffer to record at an output time falling inside the current
    // sub-step. A rate-rule target is a continuous quantity; recording it at the
    // sub-step's left endpoint would lag the true value by up to dt_max and, for
    // a purely time-driven target, manifest as spurious cross-replicate jitter
    // (the fire times — hence the sub-step grid — differ per replicate). Advance
    // the targets to the exact sample time into a scratch buffer instead. When
    // no rate rules are present this returns conc.data() unchanged, so every
    // non-rate-rule model records byte-identically. `t` is the current sub-step
    // start, where rr_deriv was snapshotted, so conc + rr_deriv·(t_sample−t) is
    // the forward-Euler value at the sample time.
    std::vector<double> rec_conc(ns);
    auto sample_conc = [&](double t_sample) -> const double * {
        if (!has_rate_rules)
            return conc.data();
        std::copy(conc.begin(), conc.end(), rec_conc.begin());
        double dt = t_sample - t;
        if (dt > 0.0)
            for (std::size_t i = 0; i < rate_rule_odes.size(); ++i)
                rec_conc[rate_rule_odes[i].target0] += rr_deriv[i] * dt;
        return rec_conc.data();
    };

    // Helper: probe currently-false triggers within (t_lo, t_hi]. Returns
    // {t_event, firing_indices}: t_event is the earliest crossing (∞ if
    // none), firing_indices are events whose t_cross is at or within
    // BISECT_EPS of t_event. Leaves model time at t_lo afterwards.
    auto probe_events_in_window = [&](double t_lo,
                                      double t_hi) -> std::pair<double, std::vector<int>> {
        double t_event = std::numeric_limits<double>::infinity();
        std::vector<int> firing_at_event;
        if (n_events == 0)
            return {t_event, firing_at_event};

        // Probe each currently-false trigger at t_hi.
        sync_state(t_hi);
        std::vector<int> potential;
        for (int ei = 0; ei < n_events; ++ei) {
            if (trigger_was_true[ei])
                continue;
            double v = eval_ref.evaluate(events[ei].trigger_expr_idx);
            if (v > 0.5) {
                potential.push_back(ei);
            }
        }

        if (!potential.empty()) {
            std::vector<double> t_cross_per(potential.size(),
                                            std::numeric_limits<double>::infinity());
            for (size_t i = 0; i < potential.size(); ++i) {
                int ei = potential[i];
                model.evaluate_functions(t_lo);
                double v_lo = eval_ref.evaluate(events[ei].trigger_expr_idx);
                if (v_lo > 0.5) {
                    t_cross_per[i] = t_lo;
                    continue;
                }
                t_cross_per[i] = bisect_trigger(ei, t_lo, t_hi);
            }
            double t_min = std::numeric_limits<double>::infinity();
            for (double tc : t_cross_per) {
                if (tc < t_min)
                    t_min = tc;
            }
            t_event = t_min;
            for (size_t i = 0; i < potential.size(); ++i) {
                if (t_cross_per[i] <= t_event + BISECT_EPS) {
                    firing_at_event.push_back(potential[i]);
                }
            }
        }

        sync_state(t_lo); // restore
        return {t_event, firing_at_event};
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

    // ─── GH #190: RR-style recompute-all + flat-scan fast loop ────────────────
    //
    // Engaged (ssa_fast_loop, decided above) when a value-specialized propensity
    // vector is available and the model is recompute-all eligible. The whole
    // vector is refilled with ONE native call, which makes the dependency-graph
    // incremental machinery (the affected-set lookup + per-affected
    // set_propensity + Fenwick update that the main loop below runs every step)
    // pure overhead. RoadRunner's direct method instead recomputes every
    // propensity each step and flat-scans a single contiguous cumulative pass
    // that both sums (a0) and selects. This branch mirrors that: one refill, one
    // flat pass for a0 + sign, one flat scan for selection — no dependency graph,
    // no Fenwick, and no per-step model.set_current_time (time() is unread for
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

            // Stuck — fast-forward all remaining samples at the frozen state.
            if (a0 <= 0.0) {
                model.update_observables(conc.data());
                for (int j = 0; j < n_obs; ++j)
                    obs_buf[j] = model.observables()[j].total;
                while (next_output < n_out) {
                    result.record(next_output, t_out[next_output], conc.data(), obs_buf.data());
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

            // Record output points strictly before the fire time. (n_func>0 only
            // when the model has time-invariant functions — time-dependent ones
            // would have set time_dependent_rates and gated this branch off — so
            // recording them here matches the main loop byte-for-byte.)
            while (next_output < n_out && t_proposed >= t_out[next_output]) {
                model.update_observables(conc.data());
                for (int j = 0; j < n_obs; ++j)
                    obs_buf[j] = model.observables()[j].total;
                result.record(next_output, t_out[next_output], conc.data(), obs_buf.data());
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
            // selection semantics as the Fenwick descent, contiguous + L1-hot).
            double target = next_u01() * a0;
            double cum = 0.0;
            int selected = nr - 1;
            for (int r = 0; r < nr; ++r) {
                double sp = a_jit[r];
                cum += (sp < 0.0) ? -sp : sp;
                if (cum >= target) {
                    selected = r;
                    break;
                }
            }

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
            auto apply_delta = [&](int si, double delta) {
                double before = conc[si];
                conc[si] += delta;
                if (before >= 0.0 && conc[si] < 0.0) {
                    ++neg_cross_count;
                    if (first_neg_species < 0)
                        first_neg_species = si;
                }
            };
            for (int ri : rxn.reactant_indices) {
                int si = ri - 1; // 1-based → 0-based
                if (si >= 0 && si < ns && !species_list[si].fixed)
                    apply_delta(si, -rstep / species_list[si].volume_factor);
            }
            for (int pi : rxn.product_indices) {
                int si = pi - 1;
                if (si >= 0 && si < ns && !species_list[si].fixed)
                    apply_delta(si, rstep / species_list[si].volume_factor);
            }

            // Advance time only (see header note: no per-step set_current_time).
            t = t_proposed;
            ++total_steps;
        }
    }

    while (next_output < n_out) {
        if (budget.active() && ++steps_since_timeout_check >= TIMEOUT_CHECK_STRIDE) {
            budget.check();
            steps_since_timeout_check = 0;
        }

        // Time-dependent rates: refresh all functions + propensities at the
        // current time so a0 reflects the rate laws at t, not at the last fire.
        if (time_dependent_rates) {
            sync_state(t);
            recompute_all_propensities();
        }

        // GH #81 — snapshot dX/dt for the rate-rule targets at the current
        // (just-synced) state. The snapshot is held constant across whatever
        // sub-step this iteration takes; integrate_rr(dt) applies it when t
        // advances. Re-snapshotting every iteration keeps the Euler step
        // current as the state evolves.
        if (has_rate_rules)
            snapshot_rr();

        // 1. Total propensity: O(log N) Fenwick prefix-sum, or O(N) flat sum.
        double a0 = sel_total();

        // 2. If total propensity is zero, the reaction system is stuck —
        //    but a time-only event trigger could still fire. Probe within
        //    (t, t_end]; if a crossing exists, advance to it and fire.
        //    Otherwise, fast-forward all remaining samples at the current
        //    state.
        if (a0 <= 0.0) {
            if (n_events > 0) {
                auto [t_event_idle, firing_at_event_idle] = probe_events_in_window(t, times.t_end);
                if (std::isfinite(t_event_idle) && t_event_idle <= times.t_end) {
                    // Record any samples strictly before t_event_idle.
                    while (next_output < n_out && t_event_idle > t_out[next_output] &&
                           t_out[next_output] < t_event_idle - SAMPLE_EVENT_TOL) {
                        const double *cc = sample_conc(t_out[next_output]);
                        model.update_observables(cc);
                        for (int j = 0; j < n_obs; ++j) {
                            obs_buf[j] = model.observables()[j].total;
                        }
                        result.record(next_output, t_out[next_output], cc, obs_buf.data());
                        if (n_func > 0) {
                            model.evaluate_functions(t_out[next_output]);
                            auto fvals = model.function_values();
                            result.record_expressions(next_output, fvals.data());
                        }
                        ++next_output;
                    }
                    if (next_output >= n_out)
                        break;
                    // GH #81: advance rate-rule targets over (t, t_event_idle]
                    // before firing so the event sees their integrated values.
                    integrate_rr(t_event_idle - t);
                    t = t_event_idle;
                    sync_state(t);
                    firing_scratch.clear();
                    std::vector<int> &firing = firing_scratch;
                    for (int ei : firing_at_event_idle) {
                        double v = eval_ref.evaluate(events[ei].trigger_expr_idx);
                        if (v > 0.5 && !trigger_was_true[ei]) {
                            firing.push_back(ei);
                        }
                    }
                    bool fired = process_firing_batch(t, firing);
                    for (int ei = 0; ei < n_events; ++ei) {
                        double v = eval_ref.evaluate(events[ei].trigger_expr_idx);
                        trigger_was_true[ei] = (v > 0.5);
                    }
                    if (fired) {
                        recompute_all_propensities();
                    }
                    model.set_current_time(t);
                    continue;
                }
            }
            // Time-dependent rates: a zero total propensity now does NOT mean
            // the system is permanently stuck — a time-varying rate (e.g. a
            // synthesis flux gated by an assignment rule that is 0 at t_start)
            // can lift off later. Advance one capped sub-step, recording any
            // samples it crosses at the current (frozen) state, then let the
            // loop refresh all functions + propensities at the new time.
            if (time_dependent_rates) {
                double t_next = std::min(t + time_dep_dt_max, times.t_end);
                while (next_output < n_out && t_out[next_output] <= t_next) {
                    const double *cc = sample_conc(t_out[next_output]);
                    model.update_observables(cc);
                    for (int j = 0; j < n_obs; ++j) {
                        obs_buf[j] = model.observables()[j].total;
                    }
                    result.record(next_output, t_out[next_output], cc, obs_buf.data());
                    if (n_func > 0) {
                        model.evaluate_functions(t_out[next_output]);
                        auto fvals = model.function_values();
                        result.record_expressions(next_output, fvals.data());
                    }
                    ++next_output;
                }
                if (next_output >= n_out)
                    break;
                // GH #81: Euler-advance the rate-rule targets across the idle
                // sub-step so a propensity that was 0 at t can lift off as the
                // continuous targets move.
                integrate_rr(t_next - t);
                t = t_next;
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

        // Time-dependent rates: cap the step so the rate law is refreshed at
        // least every dt_max. If the sampled fire time lands beyond the cap,
        // no reaction fires in this window — advance to the cap and resample
        // against the refreshed propensity next iteration (exact for the
        // piecewise-constant rate over the window by exponential memorylessness).
        bool time_dep_capped = false;
        if (time_dependent_rates && t_proposed > t + time_dep_dt_max) {
            t_proposed = t + time_dep_dt_max;
            time_dep_capped = true;
        }

        // 4. Detect event-trigger crossings within (t, t_proposed].
        //    State is piecewise-constant during τ, so a time-dependent trigger
        //    is a 1-D function of t and is well-resolved by bisection.
        //    State-dependent triggers (no time component) cannot flip during
        //    τ — those are handled post-fire below.
        auto [t_event, firing_at_event] = probe_events_in_window(t, t_proposed);

        bool event_wins = std::isfinite(t_event) && t_event < t_proposed;
        double t_advance = event_wins ? t_event : t_proposed;

        // 5. Record output points strictly before t_advance. When an event
        //    lands at (or within bisection precision of) a sample time,
        //    defer that sample so it records post-event state on the next
        //    iteration — matches ODE rootfind semantics. The tolerance is
        //    several orders wider than BISECT_EPS so a legitimate sample
        //    strictly before t_event is still recorded pre-event.
        while (next_output < n_out && t_advance >= t_out[next_output]) {
            if (event_wins && t_out[next_output] >= t_event - SAMPLE_EVENT_TOL)
                break;
            const double *cc = sample_conc(t_out[next_output]);
            model.update_observables(cc);
            for (int j = 0; j < n_obs; ++j) {
                obs_buf[j] = model.observables()[j].total;
            }
            result.record(next_output, t_out[next_output], cc, obs_buf.data());
            if (n_func > 0) {
                model.evaluate_functions(t_out[next_output]);
                auto fvals = model.function_values();
                result.record_expressions(next_output, fvals.data());
            }
            ++next_output;
        }

        if (next_output >= n_out)
            break;

        if (event_wins) {
            // Advance time to t_event, fire the batch, redraw τ. The
            // candidate reaction is NOT fired — its tentative τ assumed the
            // pre-event state and is discarded on the redraw.
            // GH #81: integrate rate-rule targets over (t, t_event] first so
            // the event's assignment RHS sees their values at the event time.
            integrate_rr(t_event - t);
            t = t_event;
            sync_state(t);
            // Confirm rising-edge status under the post-sync state and apply
            // the firing batch.
            firing_scratch.clear();
            std::vector<int> &firing = firing_scratch;
            for (int ei : firing_at_event) {
                double v = eval_ref.evaluate(events[ei].trigger_expr_idx);
                if (v > 0.5 && !trigger_was_true[ei]) {
                    firing.push_back(ei);
                }
            }
            bool fired = process_firing_batch(t, firing);
            // Update trigger_was_true to post-fire truth values.
            for (int ei = 0; ei < n_events; ++ei) {
                double v = eval_ref.evaluate(events[ei].trigger_expr_idx);
                trigger_was_true[ei] = (v > 0.5);
            }
            if (fired) {
                recompute_all_propensities();
            }
            model.set_current_time(t);
            continue;
        }

        if (time_dep_capped) {
            // No reaction fired within this capped window; advance time and let
            // the loop refresh time-dependent propensities at the new t.
            // GH #81: Euler-advance the rate-rule targets across the full cap.
            integrate_rr(t_proposed - t);
            t = t_proposed;
            model.set_current_time(t);
            continue;
        }

        // 6. Select reaction: O(log N) Fenwick descent, or O(N) flat scan.
        double r2 = next_u01() * a0;
        int selected = sel_find(r2);
        if (selected < 0)
            selected = 0;
        if (selected >= nr)
            selected = nr - 1;

        // 7. Execute reaction: update species populations
        //    PSA: scale stoichiometric coefficients by 1/λ_r
        //    Per-species volume_factor: SBML loader stores values as
        //    `amount/V_c`, so each ±1 amount fire is `±1/V_c` in storage
        //    units. Default volume_factor=1.0 → identical to ±1 fires.
        const auto &rxn = reactions[selected];
        double stoich_scale = 1.0;
        if (use_psa) {
            stoich_scale = 1.0 / scaling_factors[selected];
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

        auto apply_delta = [&](int si, double delta) {
            double before = conc[si];
            conc[si] += delta;
            if (before >= 0.0 && conc[si] < 0.0) {
                ++neg_cross_count;
                if (first_neg_species < 0)
                    first_neg_species = si;
            }
        };

        for (int ri : rxn.reactant_indices) {
            int si = ri - 1; // 1-based → 0-based
            if (si >= 0 && si < ns && !species_list[si].fixed) {
                apply_delta(si, -rstep / species_list[si].volume_factor);
            }
        }
        for (int pi : rxn.product_indices) {
            int si = pi - 1;
            if (si >= 0 && si < ns && !species_list[si].fixed) {
                apply_delta(si, rstep / species_list[si].volume_factor);
            }
        }

        // 8. Advance time unconditionally so time()/t() stay current
        // GH #81: Euler-advance the rate-rule targets over the inter-fire
        // interval (t, t_proposed] alongside the discrete fire just applied.
        integrate_rr(t_proposed - t);
        t = t_proposed;
        model.set_current_time(t);
        ++total_steps;

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

        // 10. State-dependent triggers can flip false→true after this fire.
        //     (Time-only triggers were already detected by bisection above.)
        //     Sweep all events; fire rising edges through process_firing_batch.
        if (n_events > 0) {
            firing_scratch.clear();
            std::vector<int> &firing = firing_scratch;
            for (int ei = 0; ei < n_events; ++ei) {
                double v = eval_ref.evaluate(events[ei].trigger_expr_idx);
                bool now_true = (v > 0.5);
                if (now_true && !trigger_was_true[ei]) {
                    firing.push_back(ei);
                }
                trigger_was_true[ei] = now_true;
            }
            if (process_firing_batch(t, firing)) {
                // Re-sync trigger_was_true so events that falsified their own
                // trigger can re-arm.
                for (int ei = 0; ei < n_events; ++ei) {
                    double v = eval_ref.evaluate(events[ei].trigger_expr_idx);
                    trigger_was_true[ei] = (v > 0.5);
                }
                recompute_all_propensities();
            }
        }
    }

    // ─── Write final state back to model ─────────────────────────────────────
    {
        auto &species = const_cast<std::vector<Species> &>(model.species());
        for (int i = 0; i < ns; ++i) {
            species[i].concentration = conc[i];
        }
        model.set_current_time(t);
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
        if (first_reverse_rxn >= 0) {
            const auto &names = model.species_names();
            const auto &rrx = reactions[first_reverse_rxn];
            auto join = [&](const std::vector<int> &idx1) {
                std::string s;
                for (size_t i = 0; i < idx1.size(); ++i) {
                    if (i)
                        s += " + ";
                    int si = idx1[i] - 1;
                    s += (si >= 0 && si < static_cast<int>(names.size())) ? names[si] : "?";
                }
                return s.empty() ? std::string("0") : s;
            };
            diag.first_reverse_reaction = "R" + std::to_string(rrx.index) + " (" +
                                          join(rrx.reactant_indices) + " -> " +
                                          join(rrx.product_indices) + ")";
        }
    }

    return result;
}

} // namespace bngsim
