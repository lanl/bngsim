#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "table_function.hpp"

namespace rulemonkey {

// ---- Molecule types --------------------------------------------------------

struct MoleculeTypeComponent {
  std::string name;
  std::vector<std::string> allowed_states;
};

struct MoleculeType {
  std::string id;
  std::string name;
  std::vector<MoleculeTypeComponent> components;

  int comp_index_by_name(const std::string& cname) const {
    for (int i = 0; i < static_cast<int>(components.size()); ++i)
      if (components[i].name == cname)
        return i;
    return -1;
  }

  int state_index(int comp_idx, const std::string& state) const {
    auto& allowed = components[comp_idx].allowed_states;
    for (int i = 0; i < static_cast<int>(allowed.size()); ++i)
      if (allowed[i] == state)
        return i;
    return -1;
  }

  // True iff two or more components share a name (e.g., `P(s,s)`).  A
  // pattern component `s` then maps non-uniquely onto the molecule — a
  // single pat match produces multiple embeddings — so the 2-mol-1-bond
  // fast path cannot assume a 1:1 pat→mol component mapping.
  bool has_symmetric_components() const {
    for (int i = 0; i < static_cast<int>(components.size()); ++i)
      for (int j = i + 1; j < static_cast<int>(components.size()); ++j)
        if (components[i].name == components[j].name)
          return true;
    return false;
  }
};

// ---- Pattern matching ------------------------------------------------------

enum class BondConstraint { Free, Bound, BoundTo, Wildcard };

struct PatternComponent {
  std::string name;
  int comp_type_index = -1;      // index in MoleculeType::components
  std::string required_state;    // "" = don't care
  int required_state_index = -1; // resolved index, -1 = don't care
  BondConstraint bond_constraint = BondConstraint::Wildcard;
  int bond_label = -1; // for BondTo: label to match with partner
};

struct PatternMolecule {
  std::string type_name;
  int type_index = -1; // index into Model::molecule_types
  std::vector<PatternComponent> components;
  std::string xml_id; // for Map/Operation resolution
};

// Bond pair within a pattern: two flat component indices connected by a bond.
// Flat index = sum of components in earlier PatternMolecules + local comp idx.
struct PatternBond {
  int comp_flat_a = -1;
  int comp_flat_b = -1;
};

struct Pattern {
  std::vector<PatternMolecule> molecules;
  std::vector<PatternBond> bonds;

  // For Species observables: optional stoichiometric constraint
  std::string relation; // "", "==", ">=", "<="
  int quantity = -1;

  int flat_comp_count() const {
    int n = 0;
    for (auto& m : molecules)
      n += static_cast<int>(m.components.size());
    return n;
  }

  // Get the flat component index for molecule mol_idx, local component
  // comp_idx.
  int flat_index(int mol_idx, int comp_idx) const {
    int base = 0;
    for (int i = 0; i < mol_idx; ++i)
      base += static_cast<int>(molecules[i].components.size());
    return base + comp_idx;
  }
};

// ---- Graph rewrite operations ----------------------------------------------

enum class OpType { AddBond, DeleteBond, StateChange, AddMolecule, DeleteMolecule };

struct AddMoleculeSpec {
  int type_index = -1;
  std::vector<std::pair<int, int>> comp_states; // (comp_idx, state_index)
};

struct RuleOp {
  OpType type{};

  // For AddBond / DeleteBond: flat reactant-pattern component indices
  int comp_flat_a = -1;
  int comp_flat_b = -1;

  // For AddBond to a newly added molecule: product-pattern molecule index
  // and local component index within that product molecule.  When >= 0,
  // fire_rule resolves the component from the corresponding added_mol_ids
  // entry instead of from match.comp_ids.
  int product_mol_a = -1;
  int product_comp_a = -1;
  int product_mol_b = -1;
  int product_comp_b = -1;

  // For StateChange: which flat component, new state index
  int comp_flat = -1;
  int new_state_index = -1;
  std::string new_state;     // raw string before resolution
  bool is_increment = false; // PLUS
  bool is_decrement = false; // MINUS

  // For AddMolecule
  AddMoleculeSpec add_spec;
  int add_product_mol_idx = -1; // product-pattern molecule index of the added mol

  // For DeleteBond: if true, the bond deletion may only fire when the
  // two molecules remain connected through another path (ring bond).
  // Encodes the XML attribute ensureConnected="1" from BNG, which
  // enforces the `.` (same-complex) constraint on the product side.
  bool ensure_connected = false;

  // For DeleteMolecule: which reactant-pattern molecule to delete
  int delete_pattern_mol_idx = -1;
  bool delete_connected = false; // true = delete entire species; false = molecule-only
};

// ---- Rate law --------------------------------------------------------------

enum class RateLawType { Ele, Function, MM, FunctionProduct };

enum class TfunCounterSource { None, Time, Parameter, Observable, Function };

// CONTRACT for symbolic-source fields below (`rate_expr`, `mm_kcat_expr`,
// `mm_Km_expr`, and `SpeciesInit::concentration_expr`): any `*_expr`
// capturing the un-resolved BNGL source for a parameter-derived numeric
// MUST be re-resolved in RuleMonkeySimulator::Impl::apply_overrides
// (simulator.cpp).  Otherwise set_param overrides silently fail to reach
// the engine for that field — exactly the regression that the 3.1.x
// fix-pass on apply_overrides was designed to close.

struct RateLaw {
  RateLawType type = RateLawType::Ele;
  double rate_value = 0.0;
  std::string rate_expr; // symbolic source — re-resolved in apply_overrides
  bool is_total_rate = false;

  // For Function rate law
  bool is_dynamic = false;
  std::string function_name;          // if referencing a global function
  bool is_local = false;              // has local molecule/species arguments
  bool local_arg_is_molecule = false; // true = arg bound to molecule (per-mol eval)
                                      // false = arg bound to pattern (complex-wide eval)

  // For FunctionProduct (NFsim's DOR2): the rate is the product of two
  // per-reactant local-function factors, each evaluated in the context of
  // a different tagged reactant.  `function_name`/`is_local`/
  // `local_arg_is_molecule` above describe the factor bound to reactant
  // pattern A (seed index 0); the `_b` fields below describe the factor
  // bound to reactant pattern B (seed index 1).  Realized propensity is
  // `(Σ_a w_a·f1(a)) · (Σ_b w_b·f2(b))` — see Engine::Impl DOR2 handling.
  std::string function_name_b; // second factor's local function
  bool is_local_b = false;
  bool local_arg_is_molecule_b = false;

  // For MM
  double mm_Km = 0.0;
  double mm_kcat = 0.0;
  std::string mm_kcat_expr; // symbolic source — re-resolved in apply_overrides
  std::string mm_Km_expr;   // symbolic source — re-resolved in apply_overrides

  // TFUN backing (if rate depends on a table function)
  bool uses_tfun = false;
  std::shared_ptr<TableFunction> tfun;
  TfunCounterSource tfun_counter_source = TfunCounterSource::None;
  std::string tfun_counter_name;
};

// ---- Rule ------------------------------------------------------------------

struct Rule {
  std::string id;
  std::string name;
  int molecularity = 0; // 0 (synthesis from nothing), 1, or 2
  double symmetry_factor = 1.0;
  RateLaw rate_law;

  Pattern reactant_pattern;
  Pattern product_pattern;
  std::vector<RuleOp> operations;

  // Per-ReactantPattern start indices into reactant_pattern.molecules
  // e.g., for 2 ReactantPatterns with 1 molecule each: {0, 1}
  std::vector<int> reactant_pattern_starts;

  // Reactant-to-product component map:
  // reactant flat comp index -> product flat comp index (-1 if deleted)
  std::vector<int> reactant_to_product_map;

  // True if both reactant patterns bind the same molecule type on the same
  // component type (e.g., A(a) + A(a) → A(a!1).A(a!1)).
  bool same_components = false;

  // Number of separate ProductPatterns in the XML.  If > 1 for a
  // unimolecular rule, a DeleteBond in the rule body must actually
  // separate the molecules into different complexes (the `+` between
  // product patterns requires distinct products).  When this is the
  // case and `model.block_same_complex_binding` is set, the engine
  // runs a BFS check at fire time and rejects events that leave the
  // molecules still connected (e.g., breaking one bond in a ring).
  int n_product_patterns = 0;
  std::vector<int> product_pattern_starts; // analogous to reactant_pattern_starts

  // exclude_reactants / include_reactants / exclude_products / include_products
  // Each constraint applies to one reactant or product pattern (by index).
  // Multiple patterns in a single ListOfExclude/Include are OR'd:
  //   exclude: reject if molecule matches ANY exclusion pattern
  //   include: reject if molecule matches NONE of the inclusion patterns
  struct Constraint {
    int pattern_idx{}; // which reactant/product pattern (0-based)
    Pattern pattern;   // the constraint pattern (single-molecule)
    bool is_exclude{}; // true = exclude, false = include
    bool is_product{}; // true = product constraint, false = reactant
  };
  std::vector<Constraint> constraints;
};

// ---- Observable ------------------------------------------------------------

struct Observable {
  std::string id;
  std::string name;
  std::string type; // "Molecules" or "Species"
  std::vector<Pattern> patterns;
  bool rate_dependent = false; // true if referenced by a rate law or function
};

// ---- Global function -------------------------------------------------------

struct GlobalFunction {
  std::string name;
  // Raw BNGL expression source.  Compiled into the engine's ExprTk
  // evaluator at engine init (see Engine::Impl::init_eval_layout).  An
  // empty string marks a pure-TFUN function with no wrapper expression.
  std::string expression_text;

  // Local function support
  std::vector<std::string> argument_names;         // e.g. {"z"}
  std::vector<std::string> local_observable_names; // observables referenced locally

  bool is_local() const { return !argument_names.empty(); }

  // TFUN backing
  bool is_tfun = false;
  std::shared_ptr<TableFunction> tfun;
  TfunCounterSource tfun_counter_source = TfunCounterSource::None;
  std::string tfun_counter_name;
};

// ---- Initial species -------------------------------------------------------

struct SpeciesInitMol {
  std::string type_name;
  int type_index = -1;
  std::vector<std::pair<std::string, std::string>> comp_states; // (name, state)
};

struct SpeciesInitBond {
  // Indices are (mol_local_idx, comp_name) pairs resolved to flat form
  int mol_a = -1;
  int comp_a = -1; // local component index within mol_a
  int mol_b = -1;
  int comp_b = -1;
};

struct SpeciesInit {
  std::string id;
  std::string name;
  double concentration = 0.0;
  std::string concentration_expr; // symbolic source — re-resolved in apply_overrides
  std::vector<SpeciesInitMol> molecules;
  std::vector<SpeciesInitBond> bonds;
};

// A seed species marked `Fixed="1"` in XML (BNGL `$` prefix).  Per
// BNG2 ODE semantics, such a species has d/dt = 0: its population is
// held at the initial value regardless of which reactions fire on it.
// Currently-implemented scope is restricted to single-molecule fixed
// species (no bonds within the pattern); multi-molecule complex-fixed
// is Tier-0 refused.  Multiple Fixed species of the same MoleculeType
// are also refused to avoid overlap/precedence ambiguity during
// replenishment.
struct FixedSpecies {
  int source_init_idx = -1; // index into Model::initial_species
  int mol_type_idx = -1;    // the single molecule's type
  int target_count = 0;     // clamped population (truncated from concentration)
  // Per-component required state.  Vector length = MoleculeType::components
  // size.  Each entry is the state index a matching molecule must hold
  // on that component, or -1 if any state is acceptable (typeless
  // component, or unspecified in the seed pattern).
  std::vector<int> required_comp_state;
};

// ---- Complete model --------------------------------------------------------

struct Model {
  std::vector<MoleculeType> molecule_types;
  std::unordered_map<std::string, int> molecule_type_index;

  std::vector<Rule> rules;
  std::vector<Observable> observables;

  std::unordered_map<std::string, double> parameters;
  std::vector<std::string> parameter_names_ordered;

  // Symbolic source for each declared parameter, captured at XML parse
  // time before any numeric resolution.  Used by
  // RuleMonkeySimulator::Impl::sync_parameters to recompute derived
  // parameters when set_param overrides a base parameter that other
  // parameter expressions reference (e.g., `B = 2*A` cascades when A
  // is overridden).  Keyed by parameter id; missing entries (none in
  // the current parser, but a future emitter may omit) skip the
  // cascade for that parameter and keep the parsed numeric value.
  std::unordered_map<std::string, std::string> parameter_exprs;

  std::vector<std::string> observable_names_ordered;

  std::vector<GlobalFunction> functions;
  std::unordered_map<std::string, int> function_index;

  std::unordered_map<std::string, int> observable_index;

  std::vector<SpeciesInit> initial_species;
  std::vector<FixedSpecies> fixed_species; // forward-declared; defined below

  std::string xml_path;

  // When true, bimolecular rules only fire between molecules in
  // DIFFERENT complexes (equivalent to NFsim's -bscb flag).
  //
  // Default TRUE to match strict BNGL semantics: the `+` separator in a
  // reversible rule has strict meaning in both directions.  For binding
  // (L->R), `A + B` means the reactants come from two distinct complexes
  // — intramolecular binding requires an explicit ring-closure rule of
  // the form `A(x).B(y) -> A(x!1).B(y!1)` (dot on LHS).  For unbinding
  // (R->L), `A + B` means the products end up in two distinct complexes,
  // so the bond being broken must not leave the molecules connected
  // through another path (the product molecularity check handles that
  // on the unbinding side).
  //
  // NFsim without -bscb/-cb runs in a loose mode that violates these
  // strict semantics in exchange for speed; RuleMonkey defaults to the
  // strict behaviour so it is always BNGL-correct.
  bool block_same_complex_binding = true;

  int mol_type_index(const std::string& name) const {
    auto it = molecule_type_index.find(name);
    return (it != molecule_type_index.end()) ? it->second : -1;
  }
};

} // namespace rulemonkey
