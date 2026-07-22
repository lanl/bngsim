// bngsim/include/bngsim/types.hpp — Core data structures for BNG network models
//
// All state is instance-based. No globals. No static mutable state.

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace bngsim {

// ─── Rate law types ─────────────────────────────────────────────────────────
//
// Three native rate law types. Legacy BioNetGen Sat/Hill tokens are not
// first-class native types here; the .net loader rewrites supported cases into
// synthetic Functional rate laws and emits migration guidance.
enum class RateLawType {
    Elementary,      // k * R1 * R2 * ...  (mass action)
    Functional,      // expr(observables, params, t) [* R1 * R2 * ...]
    MichaelisMenten, // kcat * E * sFree/(Km + sFree)  — tQSSA formula
};

// ─── Species ─────────────────────────────────────────────────────────────────
struct Species {
    int index;            // 1-based index from .net file
    std::string name;     // e.g. "A(b!1).B(a!1)"
    double concentration; // current concentration/count
    double initial_conc;  // initial concentration (for reset)
    bool fixed;           // boundary condition ($-prefixed species)

    // Storage→amount conversion factor used by the SSA fire step. The SBML
    // loader stores species values as `amount/V_c` regardless of hOSU, so
    // each ±1 amount fire is `±1/V_c` in storage units. Set to the species's
    // compartment volume by the SBML loader; default 1.0 leaves `.net` and
    // V=1 SBML behavior identical (fires of ±1).
    double volume_factor = 1.0;

    // Amount-valued symbol semantics (GH #75). When true, this species's
    // symbol denotes an *amount* rather than the stored concentration in
    // every rate-law / mass-action context. The engine multiplies the stored
    // value by `volume_factor` (= amount) wherever the species participates
    // in a reaction's species factor. This is the single core capability that
    // replaces the SBML loader's three independent hOSU amount-restoration
    // rewrites (the `hosu_numerator` product, the `_wrap_hosu_amounts` AST
    // rewrite, and the observable-weight reweighting): the loader now SELECTS
    // this flag per species (hasOnlySubstanceUnits=true in a V≠1 compartment)
    // instead of pre-baking `*V_c` into emitted math three different ways.
    //
    // Invariance: default false ⇒ the species factor reads the stored value
    // unchanged, so `.net` models, V=1 SBML, and every hOSU=false species are
    // byte-identical. amount = stored when volume_factor = 1.0, so V=1 hOSU
    // species are byte-identical too. Only hOSU=true species in V≠1
    // compartments move — exactly the set the loader's rewrites targeted.
    bool amount_valued = false;

    // GH #144 (case 4) — live compartment-volume index for the CROSS-COMPARTMENT
    // ODE per-species accumulation divide. A cross-compartment reaction is
    // emitted with per_species_volume_scaling=true, and compute_derivs divides
    // each species's storage derivative by its compartment volume. For an
    // hOSU=false species in a VARIABLE-volume compartment storage tracks the live
    // concentration `amount/V_live`, so the reaction term must be `/V_live`, not
    // the static `/volume_factor` (= V_static) — otherwise the post-resize
    // derivative is off by V_live/V_static (the loader's long-standing
    // "may be off by the volume ratio" warning). When >=0, this is the 0-based
    // index of the promoted compartment species whose stored value IS V_live(t);
    // compute_derivs reads conc[ode_live_volume_idx0] as the divisor for this
    // species in the per_species_volume_scaling branch. hOSU=true species keep
    // `/volume_factor` (their storage is amount/V_static — correct). Consulted
    // ONLY in that branch, so single-compartment reactions (which divide the rate
    // function by the live symbol instead) and every `.net`/static-V species are
    // unaffected. Default -1 ⇒ use volume_factor (byte-identical to pre-#144).
    int ode_live_volume_idx0 = -1;

    // GH #231 (rateOf sub-cluster 3) — when true, the SBML ``rateOf`` csymbol for
    // this species reports d(amount)/dt, NOT the stored d(conc)/dt the rateOf
    // buffer holds by default. A hasOnlySubstanceUnits=true species's symbol
    // denotes its amount, so its ``rateOf`` is the amount-rate; bngsim stores
    // concentration (amount/V_c), so d(amount)/dt = V·d(conc)/dt + conc·V̇. The
    // SBML loader sets this only for a hOSU=true species in a CONSTANT-volume
    // compartment, where V̇=0 and the amount-rate is exactly volume_factor·
    // current_derivs; refresh_rateof_derivs (and the codegen rateOf map) scale the
    // published rateOf buffer by volume_factor for these species so every
    // rate_of__<species> accessor reads the amount-rate uniformly. This holds for
    // a VARIABLE-volume compartment too (rate-rule / event-resized): the integrator
    // stores amount/V_static and rescales to live volume separately, so
    // rateof_scratch is already d(amount)/dt / V_static and ×volume_factor recovers
    // the amount-rate with no conc·V̇ correction (suite 01455/01457 constant-V,
    // 01463 rate-rule varvol). AR-compartment hOSU species stay excluded (their
    // volume is a rule function, not verified here). Default false ⇒ byte-identical
    // for .net, V=1, and every hOSU=false species (volume_factor=1 also a no-op).
    bool report_rateof_amount = false;

    // Reported in the trajectory output? (GH #71) When false this species is a
    // genuine integrator state variable — it has an ODE slot, observable, and
    // participates in the RHS/Jacobian exactly like any other species — but is
    // *projected out* of the user-facing trajectory columns (Result.species /
    // species_names, .cdat). The SBML loader sets this false for an SBML
    // *parameter* or *compartment* that is mutated only by an event assignment:
    // such a symbol must be promoted to a species to carry per-trajectory state
    // the engine can write, but it is not a floating species and RoadRunner does
    // not emit it as a trajectory column. Default true keeps `.net`, every
    // ordinary SBML species, and rate-rule-promoted parameters byte-identical
    // (rate-rule targets ARE reported — RoadRunner reports them too).
    bool reported = true;
};

// ─── Parameter ───────────────────────────────────────────────────────────────
struct Parameter {
    int index;              // 1-based index
    std::string name;       // e.g. "kf", "kon"
    double value;           // current numeric value
    std::string expression; // original expression string (empty if constant)
    bool is_expression;     // true if value was defined by expression
    int evaluator_id = -1;  // compiled expression ID (for re-evaluation after set_param)
};

// ─── Observable Group ────────────────────────────────────────────────────────
//
// Groups define observables as weighted sums of species:
//   observable = Σ factor_i * species[index_i]
//
// In .net files: "1 Obs_name  1,2*3,5" means species 1 (factor 1.0),
// species 3 (factor 2.0), species 5 (factor 1.0).
struct GroupEntry {
    int species_index; // 1-based species index
    double factor;     // multiplicity factor (default 1.0)
};

struct Observable {
    int index;        // 1-based index
    std::string name; // e.g. "Atot", "R_bound"
    std::vector<GroupEntry> entries;
    double total; // cached sum (updated before rate evaluation)
};

// ─── Reaction ────────────────────────────────────────────────────────────────
struct Reaction {
    int index;                         // 1-based index
    std::vector<int> reactant_indices; // 1-based species indices
    std::vector<int> product_indices;  // 1-based species indices
    RateLawType rate_law_type;
    std::string comment; // trailing #comment from .net

    // For ELEMENTARY: rate_law_params[0] is the rate constant parameter index
    // For FUNCTIONAL: function_name is the function reference
    // For MICHAELIS_MENTEN: rate_law_params = [kcat_index, Km_index]
    std::vector<int> rate_law_param_indices; // parameter indices
    std::string function_name;               // for Functional rate laws

    // Precomputed stoichiometry coefficient (for rate expressions with "2*kp" etc.)
    double stat_factor = 1.0;

    // BNGL convention: a Functional rate `f` on `A + B → C` has total
    // propensity `f * count(A) * count(B)` — the species factor multiplies in
    // automatically. This is what the .net loader emits and what
    // `compute_rxn_rate` does when this flag is true (the default).
    //
    // The SBML loader's unified emission registers a function whose
    // expression already includes the species factor (BNG's writeSBML emits
    // kinetic laws like `_rateLaw * S` explicitly), so multiplying by
    // species_factor again would double-count. SBML-side reactions set this
    // flag to false; the engine then treats `f` as the full propensity.
    bool apply_species_factor = true;

    // SSA-only multiplier converting an ODE-units rate (storage/time) to an
    // amount/time propensity. SBML's kinetic law value is in extent/time
    // (amount/time) per L3 spec, but our ODE rate evaluation gives storage
    // units/time = amount/(V_c·time). The SBML loader sets this to V_c (the
    // reaction's compartment volume) so SSA propensity = ODE_rate · V_c.
    // Default 1.0 leaves `.net` and V=1 SBML behavior unchanged.
    double ssa_volume_factor = 1.0;

    // Cross-compartment ODE accumulator: when true, compute_derivs divides
    // the rate by `species[si].volume_factor` per affected species instead
    // of accumulating the same rate everywhere. SBML-loader path for
    // unified Functional emission of reactions whose involved species span
    // multiple V_c values (mixed-V_s); the per-species `/V_c(s)` cannot be
    // folded into a single scalar `stat_factor`. SSA fire-step is unaffected
    // (it already divides storage delta by per-species volume_factor).
    // Default false leaves `.net` and uniform-V_s SBML behavior unchanged.
    bool per_species_volume_scaling = false;

    // GH #81 — synthetic reaction emitted from an SBML *rate rule*
    // (`dX/dt = f`). The SBML loader compiles a rate rule into a Functional
    // reaction `[] → [X]` whose rate is the rule RHS; under ODE this adds
    // `+f` to derivs[X] (the correct continuous derivative). Under exact SSA,
    // treating it as a stochastic birth/death *channel* is wrong: the target
    // X is a deterministic continuous quantity (it may even be a real-valued
    // concentration), and modeling it as integer ±1 fires injects Poisson
    // noise, rounds X to integers, and — when |f| is large — floods the
    // propensity sum so the sampler crawls. When true, the SSA/PSA loop
    // EXCLUDES this reaction from stochastic selection and instead advances
    // its target species deterministically (forward Euler) at each sub-step,
    // exactly as the CVODE path integrates `+f`. The propensities that read X
    // then follow its continuous trajectory (the loop sub-steps and refreshes
    // them). Default false leaves `.net` and every non-rate-rule reaction
    // byte-identical; the ODE path ignores this field entirely.
    bool is_rate_rule_ode = false;

    // GH #81 — live compartment-volume correction for SSA propensities in a
    // VARIABLE-volume compartment (event resize, Tier 1; rate-rule V(t),
    // Tier 2). The SSA state stores every species as `amount/V_static`
    // (V_static = load-time V_c = `ssa_volume_factor`), and the molecule
    // count `conc·V_static` is the engine-wide invariant — so the stored slot
    // is count-preserving when V moves, but each hOSU=false CONCENTRATION
    // factor in the law is then stale by `V_static/V_live`. For a mass-action
    // (monomial) law with `n_h` such factors the EXACT propensity correction
    // is `propensity *= (V_static/V_live)^(n_h-1)`: unimolecular (n_h=1) is a
    // no-op, bimolecular scales 1/V, zeroth-order synthesis scales V. The live
    // volume is read from `conc[ssa_live_volume_idx0]` — the promoted
    // compartment species the rate rule integrates / the event writes (it has
    // volume_factor=1.0, so its stored value IS V_live(t)) — and V_static is
    // `ssa_volume_factor`. `ssa_live_volume_exp` is `n_h - 1`.
    //
    // A scalar correction is exact ONLY for monomial laws (the rate function
    // is shared with ODE; the only SSA-specific lever is this multiplier), so
    // the loader sets these fields only on mass-action varvol reactions and
    // refuses non-mass-action varvol reactions under SSA. Default idx0=-1
    // leaves `.net`, static-V, and V=1 SBML byte-identical; the ODE path
    // ignores both fields. Skipped when V_live<=0 or exp==0.
    int ssa_live_volume_idx0 = -1;
    double ssa_live_volume_exp = 0.0; // n_h - 1

    // GH #144 (case 4) — CROSS-COMPARTMENT SSA live-volume correction. A reaction
    // spanning ≥2 compartments, with ≥1 variable-volume, has a SEPARATE
    // V_static/V_live ratio per varvol compartment — a product the single scalar
    // above cannot express (and in the cross-compartment emission ssa_volume_factor
    // is 1.0, so it cannot even serve as the numerator). Each term is
    // (live_idx0 = promoted compartment species whose stored value is V_live(t);
    // v_static = that compartment's load-time volume V_c; exp = m_c, the count of
    // hOSU=false reactant law-factors in compartment c). compute_rxn_rate multiplies
    // the SSA propensity by ∏_terms pow(v_static / conc[live_idx0], exp). Unlike the
    // single-compartment scalar (exp = n_f − 1, because its emission divides the
    // function by the live symbol and uses ssa_volume_factor = V_static), the
    // cross-compartment emission uses the bare law with ssa_volume_factor = 1.0, so
    // exp = m_c with no −1. Default empty ⇒ no-op (the scalar path and every
    // single-compartment / .net / static-V reaction are byte-identical); the ODE
    // path ignores this. Applied AFTER the scalar correction; a reaction uses one
    // mechanism or the other, never both.
    struct LiveVolumeTerm {
        int live_idx0;   // promoted compartment species index (stored value = V_live)
        double v_static; // V_c at load time (the propensity-correction numerator)
        double exp;      // m_c: hOSU=false reactant law-factor count in compartment c
    };
    std::vector<LiveVolumeTerm> ssa_live_volume_terms;

    // GH #81 (Tier 2) — an ODE-only reaction: excluded from SSA entirely
    // (neither fired as a channel nor integrated). The SBML loader emits the
    // #86 concentration-dilution term `-[S]·V̇/V` for an hOSU=false species in a
    // rate-rule (continuously variable-volume) compartment as a Functional
    // reaction `[] → [S]`. Under ODE that term makes the stored concentration
    // track amount/V_live (correct). Under SSA the stored value is amount/V_static
    // and the molecule COUNT (conc·V_static) must be conserved when V moves —
    // dilution does not destroy molecules, it only lowers their concentration —
    // so the dilution must NOT run (firing it as a negative-rate channel would
    // consume molecules; integrating it would shrink the count). The volume's
    // effect on reaction RATES is captured instead by the ssa_live_volume_*
    // propensity correction. Default false leaves `.net` and every ordinary
    // reaction byte-identical; the ODE path ignores this field.
    bool ode_only = false;

    // ── Pre-computed for SSA propensity evaluation ───────────────────────
    // Reactant multiplicities: (0-based species index, count) pairs.
    // Pre-computed at load time to avoid heap allocation in hot loop.
    // For A + A → B, this would be {(idx_A, 2)}.
    // For A + B → C, this would be {(idx_A, 1), (idx_B, 1)}.
    std::vector<std::pair<int, int>> reactant_multiplicities;

    // 0-based index of the rate constant parameter (first entry of
    // rate_law_param_indices, converted to 0-based). -1 if invalid.
    int rate_param_idx0 = -1;

    // For MichaelisMenten: 0-based indices + reactant species
    int mm_kcat_idx0 = -1;
    int mm_km_idx0 = -1;
    int mm_enzyme_idx0 = -1;
    int mm_substrate_idx0 = -1;
};

// ─── Function ────────────────────────────────────────────────────────────────
//
// Functions in .net files define expressions that compute variable rate parameters.
// Example: "1 _rateLaw1() k3/(K4+G)"
// The function depends on parameters and observables.
struct Function {
    int index;              // 1-based index
    std::string name;       // e.g. "_rateLaw1", "michment"
    std::string expression; // e.g. "k3/(K4+G)"
    int evaluator_id = -1;  // index into expression evaluator array
};

// ─── Stoichiometry entry (sparse) ────────────────────────────────────────────
struct StoichEntry {
    int species_index;  // 1-based
    int reaction_index; // 1-based
    double coefficient; // positive for products, negative for reactants
};

// ─── Jacobian sparsity pattern (CSC format for SUNDIALS/KLU) ─────────────────
//
// Computed at model load time from the stoichiometry matrix.
// J[i][j] ≠ 0 iff species j participates as a reactant in any reaction
// that affects species i (through stoichiometry).
// For functional rate laws, ALL species are conservatively marked as
// dependencies (since the rate expression can reference any observable).
//
// Stored in Compressed Sparse Column (CSC) format:
//   col_ptrs[j]   = index into row_indices where column j starts
//   col_ptrs[j+1] = index where column j ends
//   row_indices[k] = row index of the k-th nonzero
//
// Size: col_ptrs has n_species+1 entries, row_indices has nnz entries.
struct JacobianSparsity {
    int n;                            // matrix dimension (n_species)
    int nnz;                          // number of structural nonzeros
    std::vector<int64_t> col_ptrs;    // CSC column pointers (n+1)
    std::vector<int64_t> row_indices; // CSC row indices (nnz)
    double density;                   // nnz / (n*n), for diagnostics

    // ── Graph coloring (Curtis-Powell-Reid) for colored FD Jacobian ──────
    // Two columns share a color iff their sparsity patterns don't overlap
    // (no row index in common). This allows simultaneous perturbation of
    // all same-color columns in one RHS evaluation, reducing Jacobian cost
    // from O(N) to O(n_colors) ≈ 5–20 RHS evals for typical BNG networks.
    int n_colors = 0;                           // chromatic number (0 = not computed)
    std::vector<int> colors;                    // color assignment per column (size n)
    std::vector<std::vector<int>> color_groups; // columns grouped by color (size n_colors)

    bool empty() const { return n == 0; }
    bool has_coloring() const { return n_colors > 0 && !colors.empty(); }
};

// ─── Analytical Jacobian (pre-computed structure) ────────────────────────────
//
// For Elementary mass-action reactions v_r = k_r * sf * ∏_j x_j^{m_j}, the
// partial derivative ∂v_r/∂x_j = k_r * sf * m_j * x_j^{m_j-1} * ∏_{i≠j} x_i^{m_i}
// can be computed analytically. The Jacobian entry is:
//   J[i][j] = Σ_r S[i][r] * ∂v_r/∂x_j
// where S[i][r] is the net stoichiometric coefficient.
//
// This structure pre-computes ALL lookups at model load time so that Jacobian
// evaluation is a single O(nnz) pass with zero hash-map lookups or searches.
// For egfr_net (356 sp, 3749 rxn): ~30K operations vs O(N)=356 RHS evals for FD.

struct AnalyticalJacobianData {

    /// One "other reactant" entry: species and its multiplicity.
    struct OtherReactant {
        int species_idx;  // 0-based
        int multiplicity; // m_i (how many times this species appears)
    };

    /// Per unique reactant species j in a reaction: where to accumulate J entries.
    struct PerReactant {
        int species_idx;  // j (0-based) — the differentiation variable
        int multiplicity; // m_j

        // For each affected species i: (CSC data index, stoich coefficient)
        // J[i][j] += stoich * ∂v_r/∂x_j at these CSC positions.
        std::vector<std::pair<int64_t, double>> affected;

        // Other reactant species (excluding j) for computing the product.
        // Empty for unimolecular reactions.
        std::vector<OtherReactant> others;
    };

    /// Pre-computed terms for one reaction.
    struct ReactionTerms {
        // Defaults make a value-initialized ReactionTerms an inert placeholder
        // (rate_param_idx0 < 0 ⇒ the Elementary scatter skips it). Functional /
        // MM reactions push such a placeholder to keep ajd.reactions aligned
        // one-per-reaction; their contribution comes from the functional path.
        int rate_param_idx0 = -1; // 0-based parameter index for rate constant
        double stat_factor = 1.0; // statistical factor (e.g., 0.5 for A+A→)
        // Amount-valued reactant product (GH #75): ∏_{reactants} V_c^{m_i} over
        // amount_valued reactants (1.0 for hOSU=false / V=1). The RHS species
        // factor reads each amount_valued reactant as `x·V_c`, which introduces
        // this constant multiplier on the rate v_r and hence the same constant
        // on every ∂v_r/∂x_j. Folding it here keeps the analytical Jacobian
        // consistent with `compute_species_factor_ode`. Default 1.0 leaves
        // `.net` and V=1 SBML Jacobians byte-identical.
        double amount_factor = 1.0;
        std::vector<PerReactant> reactants; // one per unique reactant species
    };

    std::vector<ReactionTerms>
        reactions; // one per reaction (Elementary terms; empty placeholder for Functional/MM)

    /// Michaelis–Menten (tQSSA) closed-form Jacobian term (GH #76 task 3). For
    /// `rate = kcat·stat·sFree·E/(Km+sFree)` with
    /// `sFree = ½·((S-Km-E) + √((S-Km-E)² + 4·Km·S))` (clamped ≥0), ∂rate/∂E and
    /// ∂rate/∂S are emitted in closed form (the chain rule through sFree done
    /// analytically), exactly matching compute_rxn_rate's MM formula, and
    /// scattered by net stoichiometry. This makes MM models analytically
    /// covered (they previously cleared `available` and ran on finite
    /// differences). Like the Elementary terms, the closed form is trusted (it is
    /// the exact derivative of the engine's own rate law — validated against both
    /// central FD and sympy in the test suite), so it carries no per-step
    /// self-check.
    struct MMTerm {
        int e_idx;           // 0-based enzyme species (first reactant)
        int s_idx;           // 0-based substrate species (second reactant)
        int kcat_param_idx0; // 0-based kcat parameter
        int km_param_idx0;   // 0-based Km parameter
        double stat_factor;
        // Affected rows i for the E and S columns: (CSC data index, net stoich
        // coeff). J[i][E] += coeff·∂rate/∂E and J[i][S] += coeff·∂rate/∂S over the
        // reaction's net-stoichiometric species.
        std::vector<std::pair<int64_t, double>> e_affected;
        std::vector<std::pair<int64_t, double>> s_affected;
    };
    std::vector<MMTerm> mm_reactions;

    // Structural availability of the *shared* (Elementary) analytical Jacobian:
    // true iff the sparsity is non-empty and no reaction is of a type the
    // engine cannot cover analytically. Functional reactions ARE coverable —
    // their derivative expressions are supplied per-instance after build via
    // NetworkModel::set_functional_jacobian (GH #76) — so they do NOT clear
    // this flag; instead they are counted in n_functional and the model-level
    // predicate NetworkModel::analytical_jacobian_complete() additionally
    // requires the per-instance functional terms to be populated.
    //
    // For all-Elementary models n_functional == 0 and the analytical Jacobian
    // is complete the moment build() finishes — byte-identical to pre-#76.
    bool available = false;
    int n_functional = 0; // # of Functional reactions needing per-instance derivative terms
};

// ─── Functional analytical Jacobian (per-instance, GH #76) ───────────────────
//
// Symbolically-derived ∂(rate_r)/∂x_j for Functional rate laws. The derivative
// expressions are produced by the Python sympy core (bngsim._jacobian),
// compiled into THIS model instance's ExprTk evaluator (so the ids are
// instance-local — clone() recompiles them), and evaluated in the CVODE
// Jacobian callback after update_observables()/evaluate_functions() have synced
// observables/functions to the callback's state.
//
// The expressions are written in observable / parameter / time symbols — the
// symbols ExprTk keeps live during integration (species variables are stale) —
// so evaluating them yields the correct ∂func/∂x_j at the current state.
struct FunctionalJacobianData {
    // SBML / apply_species_factor==false path (per-species granularity): the
    // Python core has already chain-ruled ∂func/∂obs_k through each
    // observable's species group into one expression per dependent species.
    struct SpeciesTerm {
        int species_idx;   // j (0-based) — the differentiation variable / column
        int deriv_eval_id; // ExprTk id of the stat-free ∂func/∂x_j expression
        // One affected row i of this column j. coeff folds the reaction
        // stat_factor and the net stoichiometric coefficient (±1) but NOT the
        // volume divide (GH #171): the per-species accumulation divide is applied
        // at EVAL time so a variable-volume compartment uses the LIVE volume
        // conc[live_idx] rather than a baked-in static V. For a static-volume /
        // non-varvol row live_idx = -1 and the divide is by static_divisor
        // (= volume_factor for a per_species_volume_scaling row, else 1.0 — a
        // no-op that is byte-identical to the pre-#171 folded coeff). Scatter:
        //   J[i][j] += coeff * eval(deriv_eval_id) / divisor,
        //   divisor = (live_idx>=0 && conc[live_idx]>0) ? conc[live_idx]
        //                                               : static_divisor.
        struct Affected {
            int64_t csc;           // CSC data index of (row i, col j)
            double coeff;          // stat_factor · net_stoich (volume UNFOLDED)
            int live_idx;          // ode_live_volume_idx0 of row i, or -1
            double static_divisor; // volume_factor(i) [psvs] or 1.0
        };
        // Mirrors the Elementary PerReactant::affected layout so both callbacks
        // share the CSC→row scatter.
        std::vector<Affected> affected;
    };
    std::vector<SpeciesTerm> species_terms;

    // GH #171 — the new ∂/∂V_live column for a cross-compartment variable-volume
    // (per_species_volume_scaling) reaction. The per-species accumulation divide
    //   dSᵢ/dt = (stat·netstoichᵢ·func)/V_live
    // gains a Jacobian column at the promoted compartment species
    // V_live = conc[live_idx]:
    //   ∂(dSᵢ/dt)/∂V_live = −(stat·netstoichᵢ·func)/V_live² = −(varvol RHS of i)/V_live.
    // func is read at eval time from the reaction's bound rate parameter
    // (func_param_idx0), exactly as the per-observable path reads it, so the RHS
    // and the Jacobian share one identical func value. One term per
    // per_species_volume_scaling reaction that has ≥1 live-volume affected row.
    // A row whose live volume is ≤0 at eval used the static divisor in the RHS
    // (constant in V), so its ∂/∂V is 0 and it is simply skipped.
    struct VolumeColumnTerm {
        int func_param_idx0 = -1; // reaction's bound rate param → func value
        struct Entry {
            int64_t csc;  // CSC data index of (row i, col live_idx)
            double coeff; // stat_factor · net_stoich (volume UNFOLDED)
            int live_idx; // col = ode_live_volume_idx0 of row i (>=0)
        };
        std::vector<Entry> entries;
    };
    std::vector<VolumeColumnTerm> volume_terms;

    // .net / apply_species_factor==true path (per-observable granularity, GH #76
    // task 2). For a Functional rate `rate = func(observables) · ∏_reactants x`,
    // the column derivative is the product rule
    //   ∂rate/∂x_j = (∂func/∂x_j)·∏R + func·∂(∏R)/∂x_j,
    // with the chain rule ∂func/∂x_j = Σ_k (∂func/∂obs_k)·(∂obs_k/∂x_j). The
    // Python core emits ∂func/∂obs_k (observable symbols, live during
    // integration); the C++ scatter chain-rules through each observable's
    // species group and applies the mass-action product rule for the species
    // factor — reusing the Elementary ∂(∏R)/∂x_j machinery. `func` itself is read
    // from the reaction's bound rate parameter (evaluate_functions refreshes it),
    // so the RHS and the Jacobian share an identical func value.
    struct ObservableTerm {
        // ∂func/∂obs_k for each observable k the rate depends on (index = k).
        std::vector<int> dfunc_dobs_eval_ids; // ExprTk ids of ∂func/∂obs_k

        // One scatter column j: a species that is in some observable group of the
        // rate's observables (term A), and/or a reactant (term B, product rule).
        struct Column {
            int species_j; // j (0-based) — the differentiation variable / column
            // Term A: ∂func/∂x_j = Σ over (k, g_kj) of g_kj · eval(dfunc_dobs[k]),
            // where g_kj = ∂obs_k/∂x_j folds the observable group factor and a V_j
            // multiplier for an amount-valued species.
            std::vector<std::pair<int, double>> a_terms; // (k index, g_kj)
            // Term B: present iff j is a reactant. ∂(∏R)/∂x_j is assembled from
            // mult_j + the reaction's reactant multiset (below).
            bool is_reactant = false;
            int mult_j = 0; // multiplicity of j in the reactant multiset
            // Affected rows i: (CSC data index, coeff) where coeff folds the
            // reaction stat_factor, the net stoichiometric coefficient (±1), and a
            // 1/V_i divide when per_species_volume_scaling. J[i][j] += coeff ·
            // ((∂func/∂x_j)·∏R + func·∂(∏R)/∂x_j). Mirrors SpeciesTerm::affected.
            std::vector<std::pair<int64_t, double>> affected;
        };
        std::vector<Column> columns;

        // Reactant multiset (0-based species, multiplicity) for ∏R and ∂(∏R)/∂x_j.
        std::vector<std::pair<int, int>> reactants;
        // The reaction's bound rate parameter index (= func value after
        // evaluate_functions). -1 ⇒ no resolvable func source (skip the term).
        int func_param_idx0 = -1;
    };
    std::vector<ObservableTerm> observable_terms;

    // True once set_functional_jacobian has successfully populated terms for
    // every Functional reaction. Until then the model uses the FD Jacobian.
    bool populated = false;
};

// ─── Context handed to the Python differentiator (GH #76) ────────────────────
//
// Assembled by NetworkModel::functional_jacobian_context() from the built
// model and converted to a Python dict at the binding layer. The Python side
// (bngsim._jacobian) differentiates each Functional rate law and returns
// derivative expressions via set_functional_jacobian. The C++ engine never
// calls Python — Python drives this read→differentiate→write round-trip.
struct FunctionalJacobianContext {
    struct Rxn {
        int rxn_idx;                    // 0-based reaction index
        std::string rate_expr;          // the reaction's rate-law (function) expression
        bool apply_species_factor;      // false: per-species (SBML); true: per-observable (.net)
        std::vector<int> reactant_idx0; // 0-based reactant species
        std::vector<int> product_idx0;  // 0-based product species
        double stat_factor;
        bool per_species_volume_scaling;
    };
    std::vector<Rxn> functional_reactions;
    // name -> expression, for every user function (inlining source).
    std::vector<std::pair<std::string, std::string>> function_map;
    // observable name -> [(species_idx0, factor)] group.
    std::vector<std::pair<std::string, std::vector<std::pair<int, double>>>> observables;
    // (amount_valued, volume_factor) per species_idx0.
    std::vector<std::pair<bool, double>> species_meta;
    // parameter names safe to treat as constants (not function-bound).
    std::vector<std::string> constant_names;
};

// Input to NetworkModel::set_functional_jacobian: per Functional reaction, the
// derivative expressions the Python core produced.
struct FunctionalJacobianInput {
    int rxn_idx;         // 0-based reaction index
    bool per_observable; // false: targets are species_idx0; true: observable_idx0
    // (target_idx0, derivative ExprTk string). target is a species index when
    // per_observable is false, else an observable index.
    std::vector<std::pair<int, std::string>> deriv_terms;
};

// ─── Conservation laws ───────────────────────────────────────────────────────
//
// Detected via rank-revealing Gaussian elimination of the stoichiometry matrix.
// Each conservation law is a linear combination of species that remains constant
// during the dynamics: Σ L[k,i] * y[i] = const_k for all time.
//
// Used by the reduced-space Newton solver (steady_state.cpp) to handle models
// where conservation laws make the full Jacobian singular. The reduced system
// has N - n_laws independent equations; dependent species are reconstructed
// from the conservation constraints after convergence.
struct ConservationLaws {
    int n_laws = 0;    // number of conservation laws
    int n_species = 0; // total species count

    // Each law: coefficients[k] has n_species entries (the k-th row of L).
    // Σ coefficients[k][i] * y[i] = constants[k]
    std::vector<std::vector<double>> coefficients; // n_laws × n_species

    // Dependent species index (0-based) for each law — the species eliminated
    // in the reduced system. Chosen as the last pivot during row reduction.
    std::vector<int> dependent; // size n_laws

    // Independent species indices (0-based) — the free variables in the
    // reduced system. Size = n_species - n_laws.
    std::vector<int> independent;

    // Conservation constants computed from initial conditions.
    // constants[k] = Σ coefficients[k][i] * y0[i]
    std::vector<double> constants; // size n_laws

    bool empty() const { return n_laws == 0; }
};

// ─── Event ───────────────────────────────────────────────────────────────────
//
// Discrete state assignments triggered by boolean conditions.
// Used for drug dosing, cell division, threshold switches in SBML models.
// Events fire when a trigger expression transitions from false→true.
// Supported by ODE (CVODE rootfinding) and SSA engines.
struct Event {
    std::string id;                               // unique identifier
    int trigger_expr_idx = -1;                    // index into evaluator for trigger expression
    std::vector<std::pair<int, int>> assignments; // (species_idx_0based, value_expr_idx)
    // GH #81 (Tier 1) — per-assignment "apply under ODE only" flag, parallel
    // to `assignments`. The SBML loader injects a per-species concentration
    // rescale `s := s·V_old/V_new` when an event resizes a compartment: under
    // ODE the stored value is the true concentration `amount/V_live`, so the
    // rescale preserves amount; under SSA the stored value is `amount/V_static`
    // and the count `conc·V_static` must be preserved unchanged, so the rescale
    // must NOT run (it would corrupt counts). Entries are true only for those
    // injected rescales. Empty (the default) ⇒ every assignment applies in both
    // modes — byte-identical for `.net` and every non-resize event.
    std::vector<bool> assignment_ode_only;
    double delay = 0.0;         // delay (used iff delay_expr_idx == -1)
    int delay_expr_idx = -1;    // optional compiled delay expression
    int priority = 0;           // static fallback priority
    int priority_expr_idx = -1; // optional compiled priority expression
    bool persistent = true;     // SBML L3: trigger must stay true through delay
    bool initial_value =
        true; // trigger state at t=0 (true = treat as already true → no fire at t=0)
    bool use_values_from_trigger_time = true; // SBML L3: snapshot RHS at trigger time
};

// ─── Sensitivity options ─────────────────────────────────────────────────────
//
// When sensitivity_params is non-empty, CVODES computes dY/dp alongside the
// ODE integration. This enables gradient-based optimization and Fisher
// Information Matrix computation.
//
// ic_species_names requests forward sensitivity with respect to a species's
// initial concentration: dY(t)/dY_k(0). The variational ODE is the same as
// for parameter sens but with df/dp ≡ 0 and the seed yS_k(0) = e_k. Requires
// the codegen sens RHS path (the codegen .so dispatches df/dp = 0 via a
// sentinel parameter index, so no codegen change is needed; but the CVODES
// internal-FD fallback can't perturb a non-existent parameter).
struct SensitivityOptions {
    std::vector<std::string> param_names;      // which params to differentiate w.r.t.
    std::vector<std::string> ic_species_names; // species names for IC sens
    std::string method = "staggered";          // "simultaneous" or "staggered"
    bool error_control = true;                 // include sensitivities in error test
};

// --- Steady-state options -----------------------------------------------------
//
// Steady-state solver. Default method "integration" runs CVODE with early
// termination on the BNG2.pl parity criterion ||f(y)||_2 / n_species < tol
// (run_network -c). method "newton" runs the two-tier integrate-first solver
// (GH #27): the same CVODE burst carries the state into the physical root's
// basin, then a KINSOL Newton polish is accepted only once it is seed-stable,
// else integration continues — so "newton" is "integration" plus a polish that
// GH #28 measured at 1.4-3.9x the wall clock across six published models. What
// it buys for that is a far tighter root (residual ~1e-13 vs ~1e-9). "kinsol"
// is an accepted input alias for "newton".
struct SteadyStateOptions {
    double tol = 1e-9;     // convergence tolerance: ||f(y)||_2 / n_species < tol
    double max_time = 1e6; // max integration time for the integration path
    double rtol = 1e-8;    // CVODE relative tolerance
    double atol = 1e-8;    // CVODE absolute tolerance
    int max_steps = 20000; // max CVODE internal steps per output point
    // "integration" (default; CVODE parity early-stop), "newton" (two-tier
    // integrate-then-polish), or "kinsol" (alias for newton).
    std::string method = "integration";
    std::string jacobian = "auto"; // Jacobian strategy (same as SolverOptions)

    // Code-generated RHS shared library path
    std::string codegen_so_path;

    // Sensitivity: parameter names for steady-state sensitivity dY_ss/dp
    std::vector<std::string> sensitivity_params;
};

// --- Steady-state result ------------------------------------------------------
struct SteadyStateResult {
    std::vector<double> concentrations; // n_species steady-state values
    std::vector<std::string> species_names;
    double residual = 0.0;   // max|f(y)| at convergence
    std::string method_used; // "integration" or "newton"
    bool converged = false;
    int n_steps = 0; // CVODE steps or Newton iterations
    int n_rhs_evals = 0;

    // Steady-state sensitivity: dY_ss/dp = -J^{-1} * df/dp
    // Shape: (n_species, n_params) stored row-major
    std::vector<double> sensitivity; // empty if no sensitivity requested
    std::vector<std::string> sens_param_names;
    int n_sens_params = 0;

    // ─── Observable / function output sensitivities at steady state (GH #12) ───
    // The chain-rule projection of the species dY_ss/dp above onto the model's
    // observables and global functions, so a gradient consumer can read
    // d(observable)/dp and d(function)/dp directly instead of re-deriving the
    // output Jacobian:
    //   d(obs_j)/dp  = Σ_i (∂obs_j/∂x_i)·dY_ss_i/dp                (exact; linear groups)
    //   d(func_m)/dp = Σ_i (∂func_m/∂x_i)·dY_ss_i/dp + ∂func_m/∂p  (finite differences)
    // The function total derivative carries BOTH the state-chain term and the
    // function's explicit parameter dependence (e.g. `k3/(K4+G)` w.r.t. k3),
    // matching the CVODES codegen output-sensitivity chain rule. IC-axis output
    // sensitivities are structurally zero at a stable steady state (∂x*/∂x(0)=0)
    // and are not stored. Both blocks are row-major (n_rows × n_sens_params) and
    // empty unless sensitivity was requested AND the solve converged; the
    // function block has one row per RAW function (declaration order, including
    // the auto-generated _rateLawN intermediates), which the pybind layer
    // filters to the user-facing set that Result.expression_names exposes.
    std::vector<std::string> observable_names;  // n_observables
    std::vector<std::string> function_names;    // n_functions (raw, incl _rateLawN)
    std::vector<double> observable_sensitivity; // (n_observables × n_sens_params)
    std::vector<double> function_sensitivity;   // (n_functions × n_sens_params)
};

// ─── Solver options ──────────────────────────────────────────────────────────
struct SolverOptions {
    double rtol = 1e-8;         // relative tolerance
    double atol = 1e-8;         // absolute tolerance
    int max_steps = 20000;      // max internal steps per output point
    double max_step_size = 0.0; // 0 = no limit

    // Jacobian strategy for ODE solver:
    //   "auto"       — analytical if available, else finite-difference (default)
    //   "analytical" — force analytical Jacobian (error if not available)
    //   "fd"         — force finite-difference Jacobian (baseline)
    //   "jax"        — JAX AD Jacobian via Python callback
    std::string jacobian = "auto";

    // Force the dense direct linear solver even when the model would otherwise
    // auto-select sparse KLU (GH #102). The Jacobian *strategy* (analytical /
    // fd / jax above) is orthogonal to the linear-solver *kind* (dense vs
    // sparse): this only overrides the latter. Default false keeps the
    // size/density auto-selection (SPARSE_THRESHOLD / SPARSE_DENSITY_MAX). Set
    // true to benchmark the dense path against KLU on the same model; has no
    // effect in a build without KLU (already always dense).
    bool force_dense_linear_solver = false;

    // Force sparse KLU even when the model is below the size threshold or above
    // the density ceiling that the auto rule uses (GH #29). The mirror image of
    // force_dense_linear_solver above: that flag can only push the decision
    // toward dense, so without this one the auto rule cannot be measured
    // against its own alternative on the models it sends to dense. Bypasses
    // only the SPARSE_THRESHOLD / SPARSE_DENSITY_MAX gates — the hard
    // requirements (KLU compiled in, non-empty sparsity pattern, non-JAX
    // Jacobian) still hold, so this is a no-op in a build without KLU. Setting
    // it together with force_dense_linear_solver is an error, not a precedence
    // question; CvodeSimulator::run() rejects the pair.
    bool force_sparse_linear_solver = false;

    // JAX AD Jacobian callback.
    // If set, CVODE calls this function to fill the dense Jacobian matrix.
    // Signature: fn(t, y_ptr, jac_col_major_ptr, n_species)
    // The callback fills jac_col_major_ptr with the N×N Jacobian in
    // column-major order (matching SUNDenseMatrix layout).
    // Set from Python via pybind11. Only used when jacobian="jax".
    std::function<void(double, const double *, double *, int)> jax_jac_fn;

    // Code-generated RHS shared library path.
    // If non-empty, dlopen() this .so and use its bngsim_codegen_rhs function
    // instead of the ExprTk-based compute_derivs(). Parameters are read from
    // a runtime array (not baked as literals), so the .so is compiled once per
    // model structure and reused across all parameter evaluations.
    std::string codegen_so_path;

    // Code-generated RHS C source for the in-process MIR micro-JIT (GH #78).
    // If non-empty (and bngsim was built with BNGSIM_ENABLE_MIR), this is the
    // SAME C source the codegen emits for codegen_so_path, but JIT-compiled
    // in-process via c2mir + MIR_gen (~1-2 ms) instead of `cc -O3` + dlopen
    // (~80 ms-seconds). The resulting bngsim_codegen_rhs (and optional sens/jac)
    // plug into the identical CVODE callback seam as the dlopen path. Mutually
    // exclusive with codegen_so_path in practice; if both are set, the JIT
    // source wins.
    std::string codegen_c_source;

    // Forward sensitivity analysis via CVODES.
    // When sensitivity.param_names is non-empty, CVODES computes dY/dp
    // alongside the ODE integration. Default: empty (no sensitivities).
    SensitivityOptions sensitivity;

    // Wall-clock timeout in seconds. 0 (default) disables timeout. When > 0
    // the integration loop checks elapsed wall-clock time at each outer step
    // and at each pending-event sub-step; exceeding the limit throws
    // `bngsim::TimeoutError`. Partial results are not currently salvaged from
    // a timed-out integration.
    double timeout_seconds = 0.0;

    // Steady-state early-termination (BNG2.pl ``simulate({steady_state=>1})``
    // parity). When true, the CVODE simulator checks ||f(t,y)||_2 / n_species
    // at each output point AFTER recording it; once that norm falls below
    // ``steady_state_tol`` the integration stops and the Result is truncated
    // to only the rows actually integrated. Matches run_network -c semantics
    // (Network3 network.cpp uses NORM(derivs, n)/n < atol as the criterion).
    bool steady_state = false;
    // Tolerance for the ``steady_state`` check above. <= 0 means "use atol",
    // matching BNG2.pl which reuses its integration atol as the dx/dt cutoff.
    double steady_state_tol = 0.0;

    // Pre-equilibration / carry-over output sensitivities (GH #210). When true
    // AND the model's species state was carried over from a prior run (a
    // two-phase pre-equilibration: equilibrate-unmeasured then perturb-and-
    // measure, no reset between phases — ADR-0052), this run seeds the forward-
    // sensitivity initial conditions yS(0) from the prior phase's final
    // sensitivity matrix dx/dθ (NetworkModel::pending_sens_seed()) instead of
    // the fresh-start seed (0 for parameter columns + the IC-parameter identity).
    // That makes the measurement phase's dx/dθ correct across the boundary: the
    // phase-2 IC is the phase-1 steady state x_ss(θ), so ∂x(0)/∂θ = dx_ss/dθ,
    // not zero. Without this flag, requesting sensitivities on a carried-over
    // state RAISES (no silent wrong derivatives) — see cvode_simulator.cpp.
    // Default false (fresh-start seeding, unchanged behavior).
    bool carry_sensitivities = false;

    // Seed for random tie-breaking among simultaneous equal-priority events
    // (GH #242). SBML L3v2 §4.11.6: when several events fire at the same instant
    // with equal priority, one is chosen at random each round. bngsim implements
    // this with a per-run std::mt19937_64 seeded here (see the event-firing
    // drain in cvode_simulator.cpp). The RNG is consumed ONLY at a genuine
    // equal-priority tie, so every model without such ties is byte-identical
    // regardless of the seed, and a model WITH ties is fully reproducible for a
    // fixed seed (the PyBNF-fitting requirement). The ODE path is otherwise
    // deterministic and has no other use for this. A fixed default keeps the
    // suite / any fit reproducible out of the box; callers that want an
    // independent random realization pass their own seed.
    uint64_t event_seed = 0x9E3779B97F4A7C15ULL;
};

// ─── Time specification ──────────────────────────────────────────────────────
struct TimeSpec {
    double t_start = 0.0;
    double t_end = 100.0;
    int n_points = 101; // number of output points (including t_start)

    // Optional explicit output times. When non-empty, overrides the uniform
    // spacing derived from {t_start, t_end, n_points}. Must be sorted
    // ascending with at least 3 elements.
    std::vector<double> sample_times;

    // Resolve output times: returns sample_times if non-empty, otherwise
    // computes uniform spacing from t_start/t_end/n_points.
    std::vector<double> output_times() const {
        if (!sample_times.empty()) {
            return sample_times;
        }
        std::vector<double> t_out(n_points);
        double dt = (n_points > 1) ? (t_end - t_start) / (n_points - 1) : 0.0;
        for (int i = 0; i < n_points; ++i) {
            t_out[i] = t_start + i * dt;
        }
        return t_out;
    }

    // Effective number of output points.
    int effective_n_points() const {
        return sample_times.empty() ? n_points : static_cast<int>(sample_times.size());
    }
};

} // namespace bngsim
