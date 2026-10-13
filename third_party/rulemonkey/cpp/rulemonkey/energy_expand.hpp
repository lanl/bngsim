#pragma once

// Energy-based BNGL (eBNGL) support: load-time expansion of `Arrhenius`
// energy rules into conventional rules with pre-computed rate constants.
//
// This is a faithful port of NFsim's expansion algorithm
// (RuleWorld/nfsim src/NFcore/energyPattern.{cpp,hh}, commit c4f1bb2
// "Adds eBNGL support to NFsim", Kutuva & Faeder), which itself
// implements the Sekar (2015, PhD dissertation, Ch. 3) algorithm.
//
// Key insight (Sekar Corollary 3.3-43): only energy-pattern embeddings
// that overlap the reaction center contribute to ΔG, so each energy rule
// expands into a *finite* set of conventional rules with pre-computed
// Arrhenius rate constants — no runtime energy computation.  The context
// expansion is NFsim's; the rates are BioNetGen's (arrhenius_rate), which
// apply no RT and give phi to the rule as written (issue #107).  RuleMonkey
// already runs conventional rules with constant rates, so eBNGL is a pure
// preprocessing step at model load; the SSA loop is untouched.
//
// Scope (Phase 1, matching NFsim's current coverage): binding rules with
// exactly two reactants.  State-change energy rules and >2-reactant rules
// are refused upstream in the loader.

#include <cmath>
#include <string>
#include <vector>

namespace rulemonkey {

// ---- Parsed energy-pattern description ------------------------------------
// A lightweight mirror of an energy pattern's graph, parsed straight from
// <ListOfEnergyPatterns>.  Carries only what the expansion analysis needs:
// molecule/component identity, per-component bound/state requirements, and
// the intra-pattern bonds.

struct EpComponent {
  std::string name;             // component name, e.g. "x"
  bool is_bound = false;        // pattern requires this component bound
  std::string state_constraint; // internal state requirement, or "" if none
};

struct EpMolecule {
  std::string type_name; // e.g. "A"
  std::vector<EpComponent> components;
};

struct EnergyPatternInfo {
  std::string id;
  double energy_value = 0.0; // Gibbs free energy of formation (resolved)
  std::string energy_expr;   // unresolved source of energy_value (a parameter
                             // name or expression) — used to build a symbolic
                             // rate so set_param on an energy parameter
                             // re-resolves the expanded rates.
  std::vector<EpMolecule> molecules;

  // Bond pairs within the pattern, as (mol, comp) index quadruples.
  struct Bond {
    int mol1 = -1, comp1 = -1, mol2 = -1, comp2 = -1;
  };
  std::vector<Bond> bonds;
};

// ---- Expansion output ------------------------------------------------------

// A context constraint added to one reactant template during expansion:
// "reactant `reactant_idx`'s component `comp_name` must be bound (to
// anything) / must be free."  reactant_idx 0 = molType1, 1 = molType2 as
// passed to expand_binding().
struct EnergyContextConstraint {
  int reactant_idx = 0;
  std::string comp_name;
  bool must_be_bound = false;
};

// One expanded rule variant: a distinct combination of context constraints
// and the ΔG of making the reaction-center bond under them.  The caller
// builds one rule per variant, in the direction its XML rule writes, and
// prices it with arrhenius_rate.
struct EnergyExpandedVariant {
  std::vector<EnergyContextConstraint> constraints;
  double delta_g = 0.0; // of the binding; breaking the bond is -delta_g
  // Symbolic ΔG: the sum of the active energy patterns' `energy_expr`s (or
  // "0"), so the caller can build a set_param-re-resolvable rate expression.
  std::string delta_g_expr = "0";
};

// BioNetGen's Arrhenius rate for one rule (RateLaw.pm, evaluate_local):
// exp(-(Ea0 + c*dG)), with dG the free-energy change of the reaction the rule
// writes, c = phi for a rule as written and c = 1-phi for the reverse half of
// a reversible rule.  There is no RT: energies are in units of RT, and a
// model that works in kJ/mol divides by its own RT where it uses it, as
// BioNetGen's Models2/energy_example1.bngl does (issue #107).
inline double arrhenius_rate(double ea0, double phi, double dg, bool reverse_half) {
  double const c = reverse_half ? 1.0 - phi : phi;
  return std::exp(-(ea0 + (c * dg)));
}

// ---- Energy function -------------------------------------------------------

class EnergyFunction {
public:
  void add_pattern(EnergyPatternInfo ep) { patterns_.push_back(std::move(ep)); }
  int num_patterns() const { return static_cast<int>(patterns_.size()); }
  bool empty() const { return patterns_.empty(); }
  // Expand a two-reactant binding energy rule.
  //
  //   molType1(site1) + molType2(site2) <-> molType1(site1!1).molType2(site2!1)
  //
  // Returns one variant per context combination, with ΔG summed from the
  // energy patterns that overlap the reaction center.
  std::vector<EnergyExpandedVariant> expand_binding(const std::string& mol_type1,
                                                    const std::string& site1,
                                                    const std::string& mol_type2,
                                                    const std::string& site2) const;

private:
  std::vector<EnergyPatternInfo> patterns_;

  // Patterns containing the bond being formed/broken (molType1.site1 —
  // molType2.site2); only these differ in match count between reactant and
  // product, so only they contribute to ΔG (Sekar Corollary 3.3-43).
  std::vector<int> find_relevant_for_binding(const std::string& mol_type1, const std::string& site1,
                                             const std::string& mol_type2,
                                             const std::string& site2) const;
};

} // namespace rulemonkey
