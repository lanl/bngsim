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

  // The molecule that holds flat component `flat`, or -1 if none does.
  int mol_of_flat(int flat) const {
    int base = 0;
    for (int mi = 0; mi < static_cast<int>(molecules.size()); ++mi) {
      int const nc = static_cast<int>(molecules[mi].components.size());
      if (flat >= base && flat < base + nc)
        return mi;
      base += nc;
    }
    return -1;
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

  // For DeleteBond: true when the rule's map sends the bond's two molecules
  // into different product patterns.  The `+` between them then requires
  // the bond break to leave them in different complexes, once every
  // operation of the rule applies (issue #102).  Set by load_model from
  // reactant_to_product_map and product_pattern_starts.
  bool separates_products = false;

  // For DeleteMolecule: which reactant-pattern molecule to delete
  int delete_pattern_mol_idx = -1;
  // True for a whole-species delete: BNG2 names the ReactantPattern, not a
  // molecule, when the rule drops every molecule of the pattern without the
  // DeleteMolecules keyword (`A() -> 0`), and the matched molecule's whole
  // complex goes.  delete_pattern_mol_idx is then the pattern's first
  // molecule, which only anchors the complex.
  bool delete_connected = false;
  // A molecule delete without the keyword (BNG2 writes DeleteMolecules="0"
  // on a molecule id when the rule keeps the rest of the pattern,
  // `C(x!1).D(x!1) -> D(x)`).  Only that molecule goes, and BNG2 lets the
  // reaction happen only if what is left forms exactly the rule's product
  // patterns.  The keyword allows extra pieces holding none of the
  // molecules the rule names, but the `+` and `.` between the ones it does
  // name still hold (issue #104).
  bool delete_requires_product_count = false;
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

  // Reactant-pattern molecule a per-molecule factor is evaluated at — the
  // molecule the BNGL tag sits on (`A(b!1).B(a!1)%x`) — for the A-side and
  // B-side factor respectively.  -1 when that factor is complex-wide, the
  // DOR1 unity constant, or absent.  `local_arg_is_molecule` cannot answer
  // this: the DOR1 normalization below sets it on the unity side too.  The
  // engine seeds the tagged slot here, because it evaluates a per-molecule
  // factor at the seed (issue #88).
  int local_tag_mol = -1;
  int local_tag_mol_b = -1;

  // NFsim's DOR1: a *bimolecular* rule carrying a single local function, with
  // only one of the two reactants tagged (`S(s~0) + E()%x -> ...  lf(x)`).
  // Its propensity is `(Σ_t w_t·f(t)) · (Σ_u w_u)` over the tagged and
  // untagged reactants — i.e. a FunctionProduct whose untagged factor is the
  // constant 1.  The loader normalizes such a rule to
  // RateLawType::FunctionProduct and sets exactly one of these flags to mark
  // which side is that constant; both stay false for a genuine two-factor
  // FunctionProduct and for a unimolecular local-function rule.
  bool unity_factor_a = false;
  bool unity_factor_b = false;

  // Set when this rule's local-function chain reads something global that
  // moves during a run — a bare observable, `time`, a global function
  // built on either (issue #38).  Such a rule's per-instance rates all
  // change when nothing in any instance's own neighbourhood did, so the
  // engine's affected-molecule delta path cannot see it and the rule has
  // to be rescanned after every event.  False for a local rate built only
  // from tagged observables and constants — the shape every corpus model
  // uses — which keeps the delta path in charge there.
  bool local_rate_tracks_global = false;

  // What that chain actually reads, resolved (issue #40).  The flag above
  // answers "could this rule read a moving global?"; gating an O(N)
  // rescan needs "did one of them move?", and that needs the list, not
  // the verdict.  A bare observable is typically a volume proxy, a total
  // or a clock and is constant for the whole run, so on most models the
  // honest answer is "no" at every event and the rescan is pure waste.
  //
  // `global_dep_observables` are the observable slots the chain reads at
  // system scope, sorted and de-duplicated; comparing their values
  // against the last rescan's is exact, since a per-instance rate can
  // only move if something the rate reads moved.  The two flags below are
  // the cases with no value to compare: `time` advances every event, and
  // `global_dep_opaque` marks a dependency the §8b walk could not
  // enumerate.  Either one means "always dirty" — the unconditional
  // rescan, i.e. the pre-#40 behaviour.  All three are meaningful only
  // when `local_rate_tracks_global` is set.
  std::vector<int> global_dep_observables;
  bool global_dep_time = false;
  bool global_dep_opaque = false;

  // Set when the propensity can move with the clock alone, with no event in
  // between (issue #86): the rate law's function chain reads `time` —
  // directly, through another function, or as a TFUN counter — or reads
  // something the walk could not name.  Resolved for every Function and
  // FunctionProduct rate law, global or local, unlike the three fields
  // above.  The engine integrates such a propensity over time to place the
  // next event, where holding it at its value at the last event would be
  // exact only for a rate that does not move between events.
  bool rate_tracks_time = false;

  // Largest N for which this rate law's function chain reads the
  // `reactant_N()` placeholder described on GlobalFunction, and 0 when it
  // reads none — which is every rate law that does not use the construct.
  // The engine binds the rule's own reactant-pattern match counts into
  // those slots before evaluating the rate, so the rate function sees the
  // counts NFsim's CompositeFunction would see.
  int max_reactant_count_index = 0;

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
  // The same map for molecules: reactant-pattern molecule -> product-pattern
  // molecule, -1 if the rule deletes it.
  std::vector<int> reactant_mol_to_product_mol;

  // True if both reactant patterns bind the same molecule type on the same
  // component type (e.g., A(a) + A(a) → A(a!1).A(a!1)).
  bool same_components = false;

  // Number of separate ProductPatterns in the XML.  A DeleteBond whose two
  // molecules the map sends into different product patterns
  // (RuleOp::separates_products) must actually separate them: the `+`
  // between product patterns requires distinct products.  When
  // `model.block_same_complex_binding` is set, the engine checks this at
  // fire time, whatever the rule's molecularity, and rejects an event that
  // leaves the molecules still connected (e.g., breaking one bond in a
  // ring).  Shapes that bond-by-bond check cannot decide are listed under
  // checks_product_split below.
  int n_product_patterns = 0;
  std::vector<int> product_pattern_starts; // analogous to reactant_pattern_starts

  // The product pattern that product-pattern molecule `product_mol` is in.
  int product_pattern_of_mol(int product_mol) const {
    int pattern = -1;
    if (product_mol < 0)
      return pattern;
    for (int pi = 0; pi < static_cast<int>(product_pattern_starts.size()); ++pi)
      if (product_pattern_starts[pi] <= product_mol)
        pattern = pi;
    return pattern;
  }

  // What the `+` and `.` between product patterns can do to this rule,
  // settled at load from the map and the operations (classify_product_split
  // in simulator.cpp, issues #104 and #105).  BNG2 applies every operation,
  // then splits the result into connected components and rejects the
  // reaction if one component holds molecules of two product patterns (the
  // `+`) or one product pattern lies in two components (the `.`).
  //
  // A product pattern whose molecules nothing can bring into one complex:
  // they start in different reactant patterns, or the rule creates one,
  // and no bond the rule adds joins them (`A(s~0) + B(a) -> A(s~1).B(a)`).
  // BNG2 generates no reaction for it, and the rule never fires.
  bool products_never_join = false;
  // Molecules the rule writes into different product patterns that start
  // in one complex, where the rule deletes no bond and no molecule that
  // could part them (`A(s~0).B(t~0) -> A(s~1) + B(t~1)`).  BNG2 generates
  // no reaction for it; RM never fires it while block_same_complex_binding,
  // which enforces the `+`, is on.
  bool products_never_split = false;
  // The checks made bond by bond (RuleOp::separates_products and
  // ensure_connected) cannot decide this rule's `+` and `.`, so the engine
  // splits the post-reaction graph the way BNG2 does.  Set for a molecule
  // deleted out of a complex, for a reactant pattern its own bonds do not
  // hold together, and for one the rule's deleted bonds cut into three or
  // more pieces, whenever a `+` or `.` could then fail.
  bool checks_product_split = false;

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

// Which of a rule's reactant patterns does it TRANSFORM?
//
// A reactant pattern that no operation touches is pure context, and two
// separate results follow from that one fact:
//
//   * every embedding of it yields the identical reaction, so it is counted
//     once per matching COMPLEX rather than once per matching molecule
//     (issue #33);
//   * all of its automorphisms stabilize the reaction center, so it
//     contributes 1 to BNG2's symmetry_factor — which is what makes the
//     factor attributable to the transformed slot on an MM rule (issue #37).
//
// Both readers derive it from the flat reactant-pattern component indices the
// operations already carry (RM stores no per-reactant transformation list,
// but the mapping op -> pattern is exact), so the derivation lives here
// rather than being written twice.
//
// The classification must be conservative: an operation whose reactant
// endpoint failed to resolve at load leaves no trace on any pattern, and
// silently reading that as "context" would halve a real rule's rate.  An
// unresolved endpoint therefore clears `resolvable`, and callers must then
// treat every pattern as transformed.
struct ReactantTransforms {
  std::vector<char> transformed; // one entry per reactant pattern, 1 = touched
  // One entry per reactant pattern, 1 = the rule deletes its whole species
  // (RuleOp::delete_connected).  Such a pattern is transformed, but BNG2
  // makes the SPECIES its reaction center (RxnRule::find_reaction_center), so
  // every match of the pattern into one species is the same reaction: it is
  // counted once per matching complex, like context, and like context it
  // contributes 1 to the symmetry factor, all of its automorphisms fixing the
  // species.
  std::vector<char> species_deleted;
  bool resolvable = false;

  // Convenience for the two-slot callers.  Answers "transformed" for any
  // slot the derivation could not resolve or that does not exist.
  bool is_context(size_t slot) const {
    return resolvable && slot < transformed.size() && transformed[slot] == 0;
  }
  bool deletes_species(size_t slot) const {
    return resolvable && slot < species_deleted.size() && species_deleted[slot] != 0;
  }
};

inline ReactantTransforms reactant_pattern_transforms(const Rule& rule) {
  ReactantTransforms out;
  int const n_rp = static_cast<int>(rule.reactant_pattern_starts.size());
  int const n_rp_mols = static_cast<int>(rule.reactant_pattern.molecules.size());
  if (n_rp <= 0)
    return out;
  out.transformed.assign(static_cast<size_t>(n_rp), 0);
  out.species_deleted.assign(static_cast<size_t>(n_rp), 0);
  out.resolvable = true;

  // pattern molecule index -> reactant pattern index
  auto rp_of_mol = [&](int mi) -> int {
    int p = -1;
    for (int i = 0; i < n_rp; ++i)
      if (rule.reactant_pattern_starts[i] <= mi)
        p = i;
    return p;
  };
  // flat reactant component index -> pattern molecule index
  auto mol_of_flat = [&](int flat) -> int {
    int base = 0;
    for (int mi = 0; mi < n_rp_mols; ++mi) {
      int const nc = static_cast<int>(rule.reactant_pattern.molecules[mi].components.size());
      if (flat >= base && flat < base + nc)
        return mi;
      base += nc;
    }
    return -1;
  };
  auto touch_flat = [&](int flat) {
    int const mi = mol_of_flat(flat);
    int const p = (mi >= 0) ? rp_of_mol(mi) : -1;
    if (p < 0)
      out.resolvable = false;
    else
      out.transformed[static_cast<size_t>(p)] = 1;
  };

  for (const auto& op : rule.operations) {
    switch (op.type) {
    case OpType::StateChange:
      if (op.comp_flat >= 0)
        touch_flat(op.comp_flat);
      else
        out.resolvable = false;
      break;
    case OpType::AddBond:
    case OpType::DeleteBond:
      // Each endpoint is either a reactant component (transforms that
      // pattern) or a component of a molecule the rule synthesizes
      // (product-side only, transforms no reactant pattern).  Anything else
      // is unresolved.
      if (op.comp_flat_a >= 0)
        touch_flat(op.comp_flat_a);
      else if (op.product_mol_a < 0)
        out.resolvable = false;
      if (op.comp_flat_b >= 0)
        touch_flat(op.comp_flat_b);
      else if (op.product_mol_b < 0)
        out.resolvable = false;
      break;
    case OpType::DeleteMolecule: {
      int const mi = op.delete_pattern_mol_idx;
      int const p = (mi >= 0) ? rp_of_mol(mi) : -1;
      if (p < 0) {
        out.resolvable = false;
      } else {
        out.transformed[static_cast<size_t>(p)] = 1;
        if (op.delete_connected)
          out.species_deleted[static_cast<size_t>(p)] = 1;
      }
      break;
    }
    case OpType::AddMolecule:
      break; // product side only
    }
  }
  return out;
}

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
  std::vector<std::string> argument_names; // e.g. {"z"}

  // Observables this function references, split by scope (issue #38).  An
  // observable is LOCAL iff the expression applies it to one of this
  // function's own argument names — `NbrUp(z)` — and GLOBAL when written
  // bare — `Obs_Vol`.  BNG2's `<ListOfReferences>` marks both `Observable`
  // with nothing to tell them apart, so the split comes from parsing
  // `expression_text` (expr::classify_by_tag_application).  Both lists are
  // empty for a non-local function, where every observable is global by
  // construction.
  std::vector<std::string> local_observable_names;  // evaluated at the tagged molecule
  std::vector<std::string> global_observable_names; // read at system scope

  bool is_local() const { return !argument_names.empty(); }

  // BNGL's `reactant_N()` placeholder.  A model that wants the match count
  // of a rule's Nth reactant pattern inside a rate law declares an empty
  // function by that name (`reactant_1()` on its own line in the functions
  // block) and writes `reactant_1()*...` in the rate law.  BNG2 emits it as
  // an ordinary Function with an empty <Expression>, so nothing in the XML
  // says what it means; the name is the whole convention, and NFsim reads it
  // the same way.  Holds N (1-based) for such a placeholder and 0 for every
  // ordinary function.  The value is per rule, so the placeholder has no
  // global value of its own — see Engine::Impl::bind_reactant_counts.
  int reactant_count_index = 0;

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

  // The `<Parameter value=>` attribute verbatim, keyed by parameter id.
  // This is the LOAD-TIME source of truth: it is what NFsim reads, so
  // resolving it (rather than `expr=`) keeps RM's cold-start numbers in
  // NFsim parity even where BNG2's writer rounds `value` to fewer digits
  // than `expr` carries (e.g. `value="6.0221408e+23" expr="6.02214076e23"`).
  //
  // For BNG2-emitted XML this is a plain numeric literal.  Hand-authored
  // XML may put a symbolic expression here instead, which is why it is
  // resolved through the evaluator rather than parsed as a double.
  std::unordered_map<std::string, std::string> parameter_value_attrs;

  // The `<Parameter expr=>` attribute verbatim, keyed by parameter id —
  // the SYMBOLIC source (`LT = ((AT_nM*1e-9)*NA)*V_sim`) that BNG2 emits
  // alongside the resolved `value`.  This is what makes a `set_param`
  // override reach a *derived* parameter: `value` is already collapsed to
  // a number, so re-resolving it can never propagate an override, while
  // re-resolving `expr` can (issue #23).
  //
  // Used only by RuleMonkeySimulator::Impl::sync_parameters, and only for
  // parameters whose expr-resolved value actually moves under the active
  // overrides — see the symbolic-baseline gate there, which is what keeps
  // the no-override path bit-identical to `value`.  Entries are absent for
  // parameters that declare no `expr` (hand-authored XML), and those keep
  // the `parameter_value_attrs` behaviour.
  std::unordered_map<std::string, std::string> parameter_expr_attrs;

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
