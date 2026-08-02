// Faithful port of NFsim's Sekar energy-rule expansion.
// See energy_expand.hpp for provenance and the governing algorithm.

#include "energy_expand.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <utility>

namespace rulemonkey {

namespace {

// Internal working record for a context condition extracted from the
// conditional energy patterns.  Mirrors NFcore::ContextCondition.
struct ContextCondition {
  std::string mol_type;
  int reactant_idx = 0;
  std::string comp_name;
  std::vector<int> gated_pattern_indices; // patterns that require this bond
};

} // namespace

std::vector<int> EnergyFunction::find_relevant_for_binding(const std::string& mol_type1,
                                                           const std::string& site1,
                                                           const std::string& mol_type2,
                                                           const std::string& site2) const {
  std::vector<int> relevant;
  for (int i = 0; i < static_cast<int>(patterns_.size()); ++i) {
    const EnergyPatternInfo& ep = patterns_[i];
    for (const auto& bond : ep.bonds) {
      const EpMolecule& m1 = ep.molecules[bond.mol1];
      const EpMolecule& m2 = ep.molecules[bond.mol2];
      const std::string& c1 = m1.components[bond.comp1].name;
      const std::string& c2 = m2.components[bond.comp2].name;

      bool const match_forward =
          (m1.type_name == mol_type1 && c1 == site1 && m2.type_name == mol_type2 && c2 == site2);
      bool const match_reverse =
          (m1.type_name == mol_type2 && c1 == site2 && m2.type_name == mol_type1 && c2 == site1);
      if (match_forward || match_reverse) {
        relevant.push_back(i);
        break;
      }
    }
  }
  return relevant;
}

std::vector<EnergyExpandedVariant> EnergyFunction::expand_binding(const std::string& mol_type1,
                                                                  const std::string& site1,
                                                                  const std::string& mol_type2,
                                                                  const std::string& site2,
                                                                  double ea0, double phi) const {
  std::vector<EnergyExpandedVariant> variants;

  // Step 1: relevant patterns (contain the reaction-center bond).
  std::vector<int> const relevant = find_relevant_for_binding(mol_type1, site1, mol_type2, site2);

  if (relevant.empty()) {
    // No energy patterns overlap the reaction center → ΔG = 0 everywhere.
    EnergyExpandedVariant v;
    v.delta_g = 0.0;
    v.fwd_rate = fwd_rate(ea0, 0.0, phi);
    v.rev_rate = rev_rate(ea0, 0.0, phi);
    variants.push_back(std::move(v));
    return variants;
  }

  // Step 2: classify relevant patterns into "always" (only the reaction
  // center) and "conditional" (extra bound/stated context or a third
  // molecule type).
  std::vector<int> always_patterns;
  std::vector<int> conditional_patterns;
  for (int const pi : relevant) {
    const EnergyPatternInfo& ep = patterns_[pi];
    bool has_extra_context = false;
    for (const auto& mol : ep.molecules) {
      bool const is_reactant_type = (mol.type_name == mol_type1 || mol.type_name == mol_type2);
      if (!is_reactant_type) {
        has_extra_context = true; // a third molecule type → conditional
        break;
      }
      for (const auto& comp : mol.components) {
        if (mol.type_name == mol_type1 && comp.name == site1)
          continue; // reaction-center site
        if (mol.type_name == mol_type2 && comp.name == site2)
          continue;
        if (comp.is_bound || !comp.state_constraint.empty()) {
          has_extra_context = true;
          break;
        }
      }
      if (has_extra_context)
        break;
    }
    if (has_extra_context)
      conditional_patterns.push_back(pi);
    else
      always_patterns.push_back(pi);
  }

  // Base ΔG from always-matching patterns.
  double base_g = 0.0;
  for (int const pi : always_patterns)
    base_g += patterns_[pi].energy_value;

  // Build the symbolic ΔG (sum of the given patterns' source expressions, or
  // "0"), so the caller can synthesize a set_param-re-resolvable rate.
  auto build_dg_expr = [this](const auto& indices) -> std::string {
    std::string expr;
    for (int const pi : indices) {
      std::string term = patterns_[pi].energy_expr;
      if (term.empty())
        term = std::to_string(patterns_[pi].energy_value);
      if (!expr.empty())
        expr += "+";
      expr += "(" + term + ")";
    }
    return expr.empty() ? "0" : expr;
  };

  // Step 3: extract context conditions from conditional patterns.  A
  // condition is a bound, non-center component on a reactant-type molecule;
  // the constraint added to the template is "bound to anything" (matching
  // NFsim's addBoundComponent), so the specific partner is not recorded.
  std::map<std::pair<std::string, std::string>, ContextCondition> cond_map;
  std::vector<std::pair<std::string, std::string>> cond_order; // stable insertion order
  for (int const pi : conditional_patterns) {
    const EnergyPatternInfo& ep = patterns_[pi];
    for (const auto& mol : ep.molecules) {
      int reactant_idx = -1;
      if (mol.type_name == mol_type1)
        reactant_idx = 0;
      else if (mol.type_name == mol_type2)
        reactant_idx = 1;
      else
        continue;

      for (const auto& comp : mol.components) {
        if (reactant_idx == 0 && comp.name == site1)
          continue;
        if (reactant_idx == 1 && comp.name == site2)
          continue;
        if (!comp.is_bound)
          continue;

        auto key = std::make_pair(mol.type_name, comp.name);
        auto it = cond_map.find(key);
        if (it == cond_map.end()) {
          ContextCondition cc;
          cc.mol_type = mol.type_name;
          cc.reactant_idx = reactant_idx;
          cc.comp_name = comp.name;
          cond_map[key] = std::move(cc);
          cond_order.push_back(key);
        }
        cond_map[key].gated_pattern_indices.push_back(pi);
      }
    }
  }

  // Materialize conditions in stable insertion order (NFsim's std::map
  // iteration is by key; insertion order keeps our variant numbering
  // deterministic and independent of type-name collation — the resulting
  // rule *set* is identical either way).
  std::vector<ContextCondition> conditions;
  conditions.reserve(cond_order.size());
  for (const auto& key : cond_order) {
    ContextCondition& cc = cond_map[key];
    std::sort(cc.gated_pattern_indices.begin(), cc.gated_pattern_indices.end());
    cc.gated_pattern_indices.erase(
        std::unique(cc.gated_pattern_indices.begin(), cc.gated_pattern_indices.end()),
        cc.gated_pattern_indices.end());
    conditions.push_back(cc);
  }

  if (conditions.empty()) {
    // All relevant patterns are "always" — one variant, ΔG = base_g.
    EnergyExpandedVariant v;
    v.delta_g = base_g;
    v.delta_g_expr = build_dg_expr(always_patterns);
    v.fwd_rate = fwd_rate(ea0, base_g, phi);
    v.rev_rate = rev_rate(ea0, base_g, phi);
    variants.push_back(std::move(v));
    return variants;
  }

  // Step 4: enumerate all 2^n context combinations.
  int const n_cond = static_cast<int>(conditions.size());
  int const n_combinations = 1 << n_cond;
  for (int combo = 0; combo < n_combinations; ++combo) {
    std::set<int> active_patterns(always_patterns.begin(), always_patterns.end());

    EnergyExpandedVariant v;
    for (int ci = 0; ci < n_cond; ++ci) {
      bool const condition_met = ((combo >> ci) & 1) != 0;
      EnergyContextConstraint cc;
      cc.reactant_idx = conditions[ci].reactant_idx;
      cc.comp_name = conditions[ci].comp_name;
      cc.must_be_bound = condition_met;
      v.constraints.push_back(std::move(cc));
      if (condition_met)
        for (int const pi : conditions[ci].gated_pattern_indices)
          active_patterns.insert(pi);
    }

    double delta_g = 0.0;
    for (int const pi : active_patterns)
      delta_g += patterns_[pi].energy_value;

    v.delta_g = delta_g;
    v.delta_g_expr = build_dg_expr(active_patterns);
    v.fwd_rate = fwd_rate(ea0, delta_g, phi);
    v.rev_rate = rev_rate(ea0, delta_g, phi);
    variants.push_back(std::move(v));
  }

  return variants;
}

} // namespace rulemonkey
