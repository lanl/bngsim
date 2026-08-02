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
// Arrhenius rate constants — no runtime energy computation.  RuleMonkey
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

// One expanded rule variant: a distinct combination of context constraints,
// its ΔG, and the pre-computed forward (binding) and reverse (unbinding)
// Arrhenius rate constants.  The caller builds one binding rule (fwd_rate)
// and one unbinding rule (rev_rate) per variant.
struct EnergyExpandedVariant {
  std::vector<EnergyContextConstraint> constraints;
  double delta_g = 0.0;
  // Symbolic ΔG: the sum of the active energy patterns' `energy_expr`s (or
  // "0"), so the caller can build a set_param-re-resolvable rate expression.
  std::string delta_g_expr = "0";
  double fwd_rate = 0.0;
  double rev_rate = 0.0;
};

// ---- Energy function -------------------------------------------------------

class EnergyFunction {
public:
  explicit EnergyFunction(double rt) : rt_(rt) {}

  void add_pattern(EnergyPatternInfo ep) { patterns_.push_back(std::move(ep)); }
  int num_patterns() const { return static_cast<int>(patterns_.size()); }
  bool empty() const { return patterns_.empty(); }
  double rt() const { return rt_; }

  // Expand a two-reactant binding energy rule.
  //
  //   molType1(site1) + molType2(site2) <-> molType1(site1!1).molType2(site2!1)
  //
  // Returns one variant per context combination.  `Ea0` is the baseline
  // activation energy; `phi` is the per-rule rate-distribution parameter
  // (Arrhenius(phi, Ea0)); ΔG is summed from the energy patterns that
  // overlap the reaction center.  Rates:
  //   k_fwd = exp(-(Ea0 + phi·ΔG)      / RT)
  //   k_rev = exp(-(Ea0 + (phi-1)·ΔG)  / RT)
  std::vector<EnergyExpandedVariant> expand_binding(const std::string& mol_type1,
                                                    const std::string& site1,
                                                    const std::string& mol_type2,
                                                    const std::string& site2, double ea0,
                                                    double phi) const;

private:
  double rt_;
  std::vector<EnergyPatternInfo> patterns_;

  // Patterns containing the bond being formed/broken (molType1.site1 —
  // molType2.site2); only these differ in match count between reactant and
  // product, so only they contribute to ΔG (Sekar Corollary 3.3-43).
  std::vector<int> find_relevant_for_binding(const std::string& mol_type1, const std::string& site1,
                                             const std::string& mol_type2,
                                             const std::string& site2) const;

  double fwd_rate(double ea0, double delta_g, double phi) const {
    return std::exp(-(ea0 + (phi * delta_g)) / rt_);
  }
  double rev_rate(double ea0, double delta_g, double phi) const {
    return std::exp(-(ea0 + ((phi - 1.0) * delta_g)) / rt_);
  }
};

} // namespace rulemonkey
