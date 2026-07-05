#include "pattern_parser.hpp"

#include <algorithm>
#include <cctype>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace rulemonkey {
namespace {

std::string trim(const std::string& s) {
  size_t a = 0;
  size_t b = s.size();
  while (a < b && (std::isspace(static_cast<unsigned char>(s[a])) != 0))
    ++a;
  while (b > a && (std::isspace(static_cast<unsigned char>(s[b - 1])) != 0))
    --b;
  return s.substr(a, b - a);
}

// Split `s` on `delim` at parenthesis depth 0, trimming each piece.
// Throws (naming `ctx`) on unbalanced parentheses.
std::vector<std::string> split_top_level(const std::string& s, char delim, const std::string& ctx) {
  std::vector<std::string> out;
  std::string cur;
  int depth = 0;
  for (const char ch : s) {
    if (ch == '(') {
      ++depth;
    } else if (ch == ')') {
      if (--depth < 0)
        throw std::runtime_error("pattern parse error: unbalanced ')' in " + ctx);
    }
    if (ch == delim && depth == 0) {
      out.push_back(trim(cur));
      cur.clear();
    } else {
      cur.push_back(ch);
    }
  }
  if (depth != 0)
    throw std::runtime_error("pattern parse error: unbalanced '(' in " + ctx);
  out.push_back(trim(cur));
  return out;
}

// One component as written in the string, before resolution.
struct ParsedComp {
  std::string name;
  bool has_state = false;
  std::string state;
  bool has_bond = false;
  int bond_id = 0;
};

// One molecule as written in the string, before resolution.
struct ParsedMol {
  std::string name;
  std::vector<ParsedComp> comps;
};

// Parse `name~state!bond` (modifiers in any order) for one component.
ParsedComp parse_component(const std::string& raw, const std::string& molname) {
  const std::string text = trim(raw);
  if (text.empty())
    throw std::runtime_error("pattern parse error: empty component in molecule '" + molname + "'");

  ParsedComp comp;
  size_t pos = 0;
  while (pos < text.size() && text[pos] != '~' && text[pos] != '!')
    ++pos;
  comp.name = trim(text.substr(0, pos));
  if (comp.name.empty())
    throw std::runtime_error("pattern parse error: component without a name in molecule '" +
                             molname + "'");

  while (pos < text.size()) {
    const char marker = text[pos++];
    const size_t start = pos;
    while (pos < text.size() && text[pos] != '~' && text[pos] != '!')
      ++pos;
    const std::string value = trim(text.substr(start, pos - start));

    if (marker == '~') {
      if (comp.has_state)
        throw std::runtime_error("pattern parse error: component '" + comp.name +
                                 "' carries more than one ~state");
      if (value.empty())
        throw std::runtime_error("pattern parse error: component '" + comp.name +
                                 "' has an empty ~state");
      comp.has_state = true;
      comp.state = value;
    } else { // '!'
      if (comp.has_bond)
        throw std::runtime_error("pattern parse error: component '" + comp.name +
                                 "' carries more than one bond");
      if (value == "+" || value == "?")
        throw std::runtime_error("pattern parse error: bond wildcard '!" + value +
                                 "' on component '" + comp.name +
                                 "' — exact species patterns require numeric bond labels");
      if (value.empty())
        throw std::runtime_error("pattern parse error: component '" + comp.name +
                                 "' has an empty bond label");
      for (const char ch : value) {
        if (std::isdigit(static_cast<unsigned char>(ch)) == 0)
          throw std::runtime_error("pattern parse error: bond label '!" + value +
                                   "' on component '" + comp.name + "' is not a number");
      }
      comp.has_bond = true;
      comp.bond_id = std::stoi(value);
    }
  }
  return comp;
}

// Parse `Name(comp,comp,...)` for one molecule.  A bare `Name` with no
// parentheses is accepted (it lists zero components) and rejected later
// if the type actually has components.
ParsedMol parse_molecule(const std::string& raw) {
  const std::string text = trim(raw);
  if (text.empty())
    throw std::runtime_error("pattern parse error: empty molecule in pattern");

  ParsedMol mol;
  const size_t open = text.find('(');
  if (open == std::string::npos) {
    mol.name = text;
    return mol;
  }
  if (text.back() != ')')
    throw std::runtime_error("pattern parse error: molecule '" + text +
                             "' — component list must end with ')'");
  mol.name = trim(text.substr(0, open));
  if (mol.name.empty())
    throw std::runtime_error("pattern parse error: molecule without a name in pattern");

  const std::string inner = trim(text.substr(open + 1, text.size() - open - 2));
  if (!inner.empty()) {
    for (const auto& part : split_top_level(inner, ',', "molecule '" + mol.name + "'"))
      mol.comps.push_back(parse_component(part, mol.name));
  }
  return mol;
}

// Locate the PatternComponent at flat index `flat` within `pat`.
PatternComponent& comp_at_flat(Pattern& pat, int flat) {
  int acc = 0;
  for (auto& mol : pat.molecules) {
    const int n = static_cast<int>(mol.components.size());
    if (flat < acc + n)
      return mol.components[flat - acc];
    acc += n;
  }
  throw std::runtime_error("pattern parse error: internal flat-index overflow");
}

// Molecule index owning flat component index `flat`.
int mol_of_flat(const Pattern& pat, int flat) {
  int acc = 0;
  for (int mi = 0; mi < static_cast<int>(pat.molecules.size()); ++mi) {
    acc += static_cast<int>(pat.molecules[mi].components.size());
    if (flat < acc)
      return mi;
  }
  throw std::runtime_error("pattern parse error: internal flat-index overflow");
}

} // namespace

Pattern parse_species_pattern(const std::string& text_in, const Model& model) {
  const std::string text = trim(text_in);
  if (text.empty())
    throw std::runtime_error("pattern parse error: empty pattern string");

  // ---- 1. Tokenize -------------------------------------------------------
  std::vector<ParsedMol> pmols;
  for (const auto& part : split_top_level(text, '.', "pattern"))
    pmols.push_back(parse_molecule(part));

  // ---- 2. Resolve molecules and components against the model -------------
  Pattern pat;
  // bond label -> the flat component indices it was written on
  std::unordered_map<int, std::vector<int>> bond_endpoints;
  int flat = 0;

  for (const auto& pm : pmols) {
    const int ti = model.mol_type_index(pm.name);
    if (ti < 0)
      throw std::runtime_error("pattern parse error: unknown molecule type '" + pm.name + "'");
    const MoleculeType& mt = model.molecule_types[ti];

    // Resolve each listed component to a distinct molecule-type
    // component index.  Same-named (symmetric) components are matched
    // by occurrence: the k-th listed `r` binds the k-th type `r`.
    std::vector<int> comp_map(pm.comps.size(), -1);
    std::unordered_map<std::string, int> occurrence;
    for (size_t i = 0; i < pm.comps.size(); ++i) {
      const std::string& cname = pm.comps[i].name;
      const int occ = occurrence[cname]++;
      int found = 0;
      int idx = -1;
      for (int mi = 0; mi < static_cast<int>(mt.components.size()); ++mi) {
        if (mt.components[mi].name == cname) {
          if (found == occ) {
            idx = mi;
            break;
          }
          ++found;
        }
      }
      if (idx < 0) {
        if (found == 0)
          throw std::runtime_error("pattern parse error: molecule type '" + pm.name +
                                   "' has no component '" + cname + "'");
        throw std::runtime_error("pattern parse error: component '" + cname +
                                 "' listed more times than molecule '" + pm.name + "' has it");
      }
      comp_map[i] = idx;
    }

    // Fully-specified: every molecule-type component covered exactly once.
    if (pm.comps.size() != mt.components.size()) {
      std::vector<bool> covered(mt.components.size(), false);
      for (const int idx : comp_map)
        covered[idx] = true;
      for (int mi = 0; mi < static_cast<int>(mt.components.size()); ++mi) {
        if (!covered[mi])
          throw std::runtime_error("pattern parse error: component '" + mt.components[mi].name +
                                   "' of molecule '" + pm.name +
                                   "' is not specified — an exact species pattern must list "
                                   "every component");
      }
      throw std::runtime_error("pattern parse error: molecule '" + pm.name +
                               "' lists the wrong number of components");
    }

    PatternMolecule outm;
    outm.type_name = pm.name;
    outm.type_index = ti;
    for (size_t i = 0; i < pm.comps.size(); ++i) {
      const ParsedComp& pc = pm.comps[i];
      const int mi = comp_map[i];
      const MoleculeTypeComponent& mc = mt.components[mi];

      PatternComponent oc;
      oc.name = pc.name;
      oc.comp_type_index = mi;

      // State: a stateful component needs a concrete, allowed state; a
      // stateless component must carry none.
      if (!mc.allowed_states.empty()) {
        if (!pc.has_state)
          throw std::runtime_error("pattern parse error: component '" + pc.name +
                                   "' of molecule '" + pm.name +
                                   "' is stateful and needs an explicit ~state");
        const int si = mt.state_index(mi, pc.state);
        if (si < 0)
          throw std::runtime_error("pattern parse error: '" + pc.state +
                                   "' is not an allowed state of component '" + pc.name +
                                   "' on molecule '" + pm.name + "'");
        oc.required_state = pc.state;
        oc.required_state_index = si;
      } else if (pc.has_state) {
        throw std::runtime_error("pattern parse error: component '" + pc.name + "' of molecule '" +
                                 pm.name + "' is stateless but a ~state was given");
      }

      // Bond: numeric label now, resolved to a PatternBond below.
      if (pc.has_bond) {
        oc.bond_constraint = BondConstraint::BoundTo;
        bond_endpoints[pc.bond_id].push_back(flat + static_cast<int>(i));
      } else {
        oc.bond_constraint = BondConstraint::Free;
      }
      outm.components.push_back(std::move(oc));
    }
    flat += static_cast<int>(outm.components.size());
    pat.molecules.push_back(std::move(outm));
  }

  // ---- 3. Bonds ----------------------------------------------------------
  // Each numeric label must appear on exactly two distinct components.
  // Process labels in ascending order so pat.bonds is deterministic.
  std::vector<int> labels;
  labels.reserve(bond_endpoints.size());
  for (const auto& kv : bond_endpoints)
    labels.push_back(kv.first);
  std::sort(labels.begin(), labels.end());
  for (const int lbl : labels) {
    const std::vector<int>& eps = bond_endpoints[lbl];
    if (eps.size() != 2)
      throw std::runtime_error("pattern parse error: bond label '!" + std::to_string(lbl) +
                               "' appears on " + std::to_string(eps.size()) +
                               " component(s) — a bond must connect exactly two");
    if (eps[0] == eps[1])
      throw std::runtime_error("pattern parse error: bond label '!" + std::to_string(lbl) +
                               "' connects a component to itself");
    PatternBond pb;
    pb.comp_flat_a = eps[0];
    pb.comp_flat_b = eps[1];
    const int bidx = static_cast<int>(pat.bonds.size());
    pat.bonds.push_back(pb);
    comp_at_flat(pat, eps[0]).bond_label = bidx;
    comp_at_flat(pat, eps[1]).bond_label = bidx;
  }

  // ---- 4. Connectivity ---------------------------------------------------
  // The molecules must form one connected complex — a single species.
  const int n_mol = static_cast<int>(pat.molecules.size());
  if (n_mol > 1) {
    std::vector<int> parent(n_mol);
    for (int i = 0; i < n_mol; ++i)
      parent[i] = i;
    const std::function<int(int)> find = [&](int x) {
      while (parent[x] != x)
        x = parent[x] = parent[parent[x]];
      return x;
    };
    for (const auto& b : pat.bonds) {
      const int ra = find(mol_of_flat(pat, b.comp_flat_a));
      const int rb = find(mol_of_flat(pat, b.comp_flat_b));
      if (ra != rb)
        parent[ra] = rb;
    }
    const int root0 = find(0);
    for (int i = 1; i < n_mol; ++i) {
      if (find(i) != root0)
        throw std::runtime_error(
            "pattern parse error: the molecules do not form one connected complex — "
            "an exact species pattern must denote a single species");
    }
  }

  return pat;
}

} // namespace rulemonkey
