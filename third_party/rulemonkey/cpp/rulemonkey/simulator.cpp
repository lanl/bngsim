#include "rulemonkey/simulator.hpp"

#include "energy_expand.hpp"
#include "engine.hpp"
#include "expr_eval.hpp"
#include "model.hpp"
#include "pattern_parser.hpp"
#include "table_function.hpp"

#include "bngsim/expression.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rulemonkey {

// ===========================================================================
// String helpers
// ===========================================================================

namespace {

std::string trim(const std::string& s) {
  size_t a = 0;
  while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a])))
    ++a;
  size_t b = s.size();
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1])))
    --b;
  return s.substr(a, b - a);
}

std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

// ===========================================================================
// XML parser
// ===========================================================================

// Append the UTF-8 encoding of a Unicode code point to `out`.  Used by
// decode_xml_entities for `&#NNN;` and `&#xHH;` numeric character refs.
// Throws on values outside the valid Unicode range or on surrogate halves.
void append_utf8(std::string& out, uint32_t cp) {
  if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
    throw std::runtime_error("XML: numeric entity out of range or surrogate half: U+" +
                             std::to_string(cp));
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

std::string decode_xml_entities(std::string_view input) {
  std::string out;
  out.reserve(input.size());
  size_t i = 0;
  while (i < input.size()) {
    if (input[i] != '&') {
      out.push_back(input[i]);
      ++i;
      continue;
    }
    auto sc = input.find(';', i + 1);
    if (sc == std::string_view::npos)
      throw std::runtime_error("XML: unterminated entity");
    auto ent = input.substr(i + 1, sc - i - 1);
    if (ent == "amp")
      out.push_back('&');
    else if (ent == "lt")
      out.push_back('<');
    else if (ent == "gt")
      out.push_back('>');
    else if (ent == "apos")
      out.push_back('\'');
    else if (ent == "quot")
      out.push_back('"');
    else if (!ent.empty() && ent[0] == '#') {
      // Numeric character reference: &#NNN; (decimal) or &#xHH; (hex).
      // BNG2 doesn't currently emit these, but third-party XML
      // pipelines do (e.g. SBML output from non-BNG tools), so the
      // parser may as well handle them rather than throwing.
      if (ent.size() < 2)
        throw std::runtime_error("XML: empty numeric entity '&;'");
      uint32_t cp = 0;
      bool const hex = (ent[1] == 'x' || ent[1] == 'X');
      auto digits = ent.substr(hex ? 2 : 1);
      if (digits.empty())
        throw std::runtime_error("XML: empty numeric entity '&" + std::string(ent) + ";'");
      for (char const c : digits) {
        uint32_t d = 0;
        if (c >= '0' && c <= '9')
          d = static_cast<uint32_t>(c - '0');
        else if (hex && c >= 'a' && c <= 'f')
          d = 10 + static_cast<uint32_t>(c - 'a');
        else if (hex && c >= 'A' && c <= 'F')
          d = 10 + static_cast<uint32_t>(c - 'A');
        else
          throw std::runtime_error("XML: malformed numeric entity '&" + std::string(ent) + ";'");
        // Multiply-and-add with overflow guard so a pathological 32-digit
        // hex entity can't silently wrap.
        uint64_t const next = (static_cast<uint64_t>(cp) * (hex ? 16ULL : 10ULL)) + d;
        if (next > 0x10FFFF)
          throw std::runtime_error("XML: numeric entity out of Unicode range: '&" +
                                   std::string(ent) + ";'");
        cp = static_cast<uint32_t>(next);
      }
      append_utf8(out, cp);
    } else
      throw std::runtime_error("XML: unsupported entity '&" + std::string(ent) + ";'");
    i = sc + 1;
  }
  return out;
}

struct XmlNode {
  std::string name;
  std::unordered_map<std::string, std::string> attributes;
  std::vector<XmlNode> children;
  std::string text;
};

class XmlParser {
public:
  explicit XmlParser(std::string src) : src_(std::move(src)) {}

  XmlNode parse_document() {
    skip_prolog();
    if (done())
      fail("empty document");
    auto root = parse_element();
    skip_prolog();
    if (!done())
      fail("trailing content");
    return root;
  }

private:
  bool done() const { return pos_ >= src_.size(); }
  bool starts_with(std::string_view sv) const {
    return pos_ + sv.size() <= src_.size() && std::string_view(src_).substr(pos_, sv.size()) == sv;
  }
  [[noreturn]] void fail(const std::string& msg) const {
    throw std::runtime_error("XML parse error: " + msg + " (offset " + std::to_string(pos_) + ")");
  }
  void skip_ws() {
    while (!done() && std::isspace(static_cast<unsigned char>(src_[pos_])))
      ++pos_;
  }
  void skip_pi() {
    auto e = src_.find("?>", pos_ + 2);
    if (e == std::string::npos)
      fail("unterminated PI");
    pos_ = e + 2;
  }
  void skip_comment() {
    auto e = src_.find("-->", pos_ + 4);
    if (e == std::string::npos)
      fail("unterminated comment");
    pos_ = e + 3;
  }
  void skip_doctype() {
    auto e = src_.find('>', pos_ + 2);
    if (e == std::string::npos)
      fail("unterminated DOCTYPE");
    pos_ = e + 1;
  }
  void skip_prolog() {
    for (;;) {
      skip_ws();
      if (starts_with("<?")) {
        skip_pi();
        continue;
      }
      if (starts_with("<!--")) {
        skip_comment();
        continue;
      }
      if (starts_with("<!DOCTYPE")) {
        skip_doctype();
        continue;
      }
      break;
    }
  }
  static bool is_name_start(char c) {
    return std::isalpha(static_cast<unsigned char>(c)) || c == '_' || c == ':';
  }
  static bool is_name_char(char c) {
    return is_name_start(c) || std::isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '.';
  }
  std::string parse_name() {
    if (done() || !is_name_start(src_[pos_]))
      fail("expected name");
    auto s = pos_++;
    while (!done() && is_name_char(src_[pos_]))
      ++pos_;
    return src_.substr(s, pos_ - s);
  }
  std::string parse_attr_val() {
    if (done())
      fail("expected attribute value");
    char const q = src_[pos_];
    if (q != '"' && q != '\'')
      fail("attribute value must be quoted");
    ++pos_;
    auto s = pos_;
    while (!done() && src_[pos_] != q)
      ++pos_;
    if (done())
      fail("unterminated attribute value");
    auto raw = src_.substr(s, pos_ - s);
    ++pos_;
    return decode_xml_entities(raw);
  }
  std::string parse_text() {
    auto s = pos_;
    while (!done() && src_[pos_] != '<')
      ++pos_;
    return decode_xml_entities(std::string_view(src_).substr(s, pos_ - s));
  }
  std::string parse_cdata() {
    pos_ += 9; // skip <![CDATA[
    auto e = src_.find("]]>", pos_);
    if (e == std::string::npos)
      fail("unterminated CDATA");
    auto val = src_.substr(pos_, e - pos_);
    pos_ = e + 3;
    return val;
  }

  XmlNode parse_element() {
    if (done() || src_[pos_] != '<')
      fail("expected '<'");
    ++pos_;
    XmlNode nd;
    nd.name = parse_name();
    for (;;) {
      skip_ws();
      if (done())
        fail("unterminated tag '" + nd.name + "'");
      if (starts_with("/>")) {
        pos_ += 2;
        return nd;
      }
      if (src_[pos_] == '>') {
        ++pos_;
        break;
      }
      auto an = parse_name();
      skip_ws();
      if (done() || src_[pos_] != '=')
        fail("expected '=' after attr");
      ++pos_;
      skip_ws();
      nd.attributes[an] = parse_attr_val();
    }
    for (;;) {
      if (done())
        fail("unterminated element '" + nd.name + "'");
      if (starts_with("</")) {
        pos_ += 2;
        auto en = parse_name();
        skip_ws();
        if (done() || src_[pos_] != '>')
          fail("expected '>'");
        ++pos_;
        if (en != nd.name)
          fail("mismatched tag: expected </" + nd.name + "> got </" + en + ">");
        nd.text = trim(nd.text);
        return nd;
      }
      if (starts_with("<!--")) {
        skip_comment();
        continue;
      }
      if (starts_with("<?")) {
        skip_pi();
        continue;
      }
      if (starts_with("<![CDATA[")) {
        nd.text += parse_cdata();
        continue;
      }
      if (src_[pos_] == '<') {
        nd.children.push_back(parse_element());
        continue;
      }
      nd.text += parse_text();
    }
  }

  std::string src_;
  size_t pos_ = 0;
};

// XML helpers
const XmlNode* find_child(const XmlNode& p, const std::string& name) {
  for (auto& c : p.children)
    if (c.name == name)
      return &c;
  return nullptr;
}
std::string need_attr(const XmlNode& n, const std::string& a) {
  auto it = n.attributes.find(a);
  if (it == n.attributes.end() || it->second.empty())
    throw std::runtime_error("XML: missing attr '" + a + "' on <" + n.name + ">");
  return it->second;
}
std::string opt_attr(const XmlNode& n, const std::string& a) {
  auto it = n.attributes.find(a);
  return (it != n.attributes.end()) ? it->second : std::string{};
}

// Forward declaration for unsupported-feature scanner (defined below load_model)
std::vector<UnsupportedFeature> scan_unsupported(const XmlNode& model_node);

// ===========================================================================
// BNG XML → Model parsing
// ===========================================================================

// Resolve a parameter-derived numeric expression against an ExprTk
// evaluator, memoizing the compiled-expression id.
//
// `s` is the symbolic source captured from XML.  `ev` must already have
// every parameter the expression can reference bound as a variable
// (see Impl::build_param_evaluator and load_model's parameter loop).
// `id_cache` maps the raw expression string to its compiled id, so a
// caller re-resolving the same expression (parameter cascade,
// apply_overrides) pays the ExprTk compile cost only once.
//
// Compiled expressions are immutable; only the bound parameter values
// change.  set_param mutates those values but does NOT invalidate the
// cache — re-evaluating a compiled id picks up the new values.
//
// Throws std::runtime_error if `ev.compile()` rejects `s` (genuine
// syntax error / unknown identifier); callers that tolerate forward
// references (the cascade) wrap the call in try/catch.
double resolve_cached(const std::string& s, bngsim::ExprTkEvaluator& ev,
                      std::unordered_map<std::string, int>& id_cache) {
  if (s.empty())
    return 0.0;
  auto it = id_cache.find(s);
  int id;
  if (it != id_cache.end()) {
    id = it->second;
  } else {
    id = ev.compile(s);
    id_cache.emplace(s, id);
  }
  return ev.evaluate(id);
}

// Parse a pattern (reactant, product, or observable) from XML.
Pattern parse_pattern(const XmlNode& pat_node, const Model& model,
                      std::unordered_map<std::string, std::pair<int, int>>* id_to_flat = nullptr) {
  Pattern pat;

  // Molecules
  auto* mol_list = find_child(pat_node, "ListOfMolecules");
  if (mol_list) {
    for (auto& mn : mol_list->children) {
      if (mn.name != "Molecule")
        continue;
      PatternMolecule pm;
      pm.xml_id = need_attr(mn, "id");
      pm.type_name = need_attr(mn, "name");
      pm.type_index = model.mol_type_index(pm.type_name);
      if (pm.type_index < 0)
        throw std::runtime_error("Unknown molecule type '" + pm.type_name + "'");

      auto& mtype = model.molecule_types[pm.type_index];
      auto* comp_list = find_child(mn, "ListOfComponents");
      if (comp_list) {
        for (auto& cn : comp_list->children) {
          if (cn.name != "Component")
            continue;
          PatternComponent pc;
          pc.name = need_attr(cn, "name");
          // Find comp type index — handle symmetric components (e.g., r1, r2)
          pc.comp_type_index = mtype.comp_index_by_name(pc.name);
          if (pc.comp_type_index < 0) {
            // Try stripping trailing digits for symmetric components
            std::string base = pc.name;
            while (!base.empty() && std::isdigit(static_cast<unsigned char>(base.back())))
              base.pop_back();
            pc.comp_type_index = mtype.comp_index_by_name(base);
          }

          // State
          auto state = opt_attr(cn, "state");
          if (!state.empty()) {
            pc.required_state = state;
            if (pc.comp_type_index >= 0)
              pc.required_state_index = mtype.state_index(pc.comp_type_index, state);
          }

          // Bond constraint
          auto bonds = opt_attr(cn, "numberOfBonds");
          if (bonds == "0") {
            pc.bond_constraint = BondConstraint::Free;
          } else if (bonds == "+" || bonds == "?") {
            pc.bond_constraint = (bonds == "+") ? BondConstraint::Bound : BondConstraint::Wildcard;
          } else if (!bonds.empty()) {
            // Specific bond count — treat as Bound
            pc.bond_constraint = BondConstraint::Bound;
          }

          // Register xml_id -> flat index
          auto comp_id = opt_attr(cn, "id");
          if (id_to_flat && !comp_id.empty()) {
            int const mol_idx = static_cast<int>(pat.molecules.size());
            int const comp_idx = static_cast<int>(pm.components.size());
            (*id_to_flat)[comp_id] = {mol_idx, comp_idx};
          }

          pm.components.push_back(std::move(pc));
        }
      }

      // Register molecule xml_id
      if (id_to_flat) {
        int const mol_idx = static_cast<int>(pat.molecules.size());
        (*id_to_flat)[pm.xml_id] = {mol_idx, -1};
      }

      pat.molecules.push_back(std::move(pm));
    }
  }

  // Bonds
  auto* bond_list = find_child(pat_node, "ListOfBonds");
  if (bond_list) {
    for (auto& bn : bond_list->children) {
      if (bn.name != "Bond")
        continue;
      auto s1 = need_attr(bn, "site1");
      auto s2 = need_attr(bn, "site2");
      if (id_to_flat) {
        auto it1 = id_to_flat->find(s1);
        auto it2 = id_to_flat->find(s2);
        if (it1 != id_to_flat->end() && it2 != id_to_flat->end()) {
          PatternBond pb;
          pb.comp_flat_a = pat.flat_index(it1->second.first, it1->second.second);
          pb.comp_flat_b = pat.flat_index(it2->second.first, it2->second.second);
          pat.bonds.push_back(pb);

          // Mark components as BoundTo
          auto& c1 = pat.molecules[it1->second.first].components[it1->second.second];
          auto& c2 = pat.molecules[it2->second.first].components[it2->second.second];
          int const label = static_cast<int>(pat.bonds.size()) - 1;
          c1.bond_constraint = BondConstraint::BoundTo;
          c1.bond_label = label;
          c2.bond_constraint = BondConstraint::BoundTo;
          c2.bond_label = label;
        } else {
          // One or both bond endpoints reference a site ID that isn't in
          // the current pattern's id_to_flat map.  Previously silent —
          // a typo or a dangling cross-pattern reference would drop the
          // bond and the pattern would match too freely.  Warn loudly so
          // model authors can see what was lost; we still don't throw,
          // since BNG2 is the authoritative XML emitter and may emit
          // shapes we haven't catalogued (the warning surfaces both
          // genuine bugs and false positives for the same eyeball pass).
          auto bond_id = opt_attr(bn, "id");
          std::fprintf(stderr,
                       "Warning: parse_pattern dropped bond '%s' (site1='%s', site2='%s'): "
                       "%s%s%s did not resolve in the current pattern's component map. "
                       "Pattern will match without this bond constraint.\n",
                       bond_id.c_str(), s1.c_str(), s2.c_str(),
                       it1 == id_to_flat->end() ? "site1" : "",
                       (it1 == id_to_flat->end() && it2 == id_to_flat->end()) ? " and " : "",
                       it2 == id_to_flat->end() ? "site2" : "");
        }
      }
    }
  }

  return pat;
}

// ---- eBNGL energy-rule expansion helpers ----------------------------------
// Support for `Arrhenius` energy rate laws: at model load, each energy rule
// is expanded into a finite set of conventional rules with pre-computed rate
// constants (Sekar 2015, ported from NFsim — see energy_expand.hpp).  These
// helpers convert parsed energy patterns and synthesize the expanded binding
// / unbinding Rule structs.  The SSA loop is untouched.

// Convert a parsed RM Pattern (from an <EnergyPattern>) into the lightweight
// EnergyPatternInfo the expansion analysis consumes.
EnergyPatternInfo pattern_to_energy_info(const std::string& id, double energy,
                                         const std::string& energy_expr, const Pattern& pat) {
  EnergyPatternInfo ep;
  ep.id = id;
  ep.energy_value = energy;
  ep.energy_expr = energy_expr;
  for (const auto& pm : pat.molecules) {
    EpMolecule m;
    m.type_name = pm.type_name;
    for (const auto& pc : pm.components) {
      EpComponent c;
      c.name = pc.name;
      // A component is "bound" for expansion purposes if the pattern pins it
      // to a specific partner (BoundTo) or to any partner (Bound / `!+`).
      c.is_bound = (pc.bond_constraint == BondConstraint::Bound ||
                    pc.bond_constraint == BondConstraint::BoundTo);
      c.state_constraint = pc.required_state;
      m.components.push_back(std::move(c));
    }
    ep.molecules.push_back(std::move(m));
  }
  auto locate = [&pat](int flat, int& mol, int& comp) {
    int running = 0;
    for (int mi = 0; mi < static_cast<int>(pat.molecules.size()); ++mi) {
      int const nc = static_cast<int>(pat.molecules[mi].components.size());
      if (flat >= running && flat < running + nc) {
        mol = mi;
        comp = flat - running;
        return;
      }
      running += nc;
    }
    mol = -1;
    comp = -1;
  };
  for (const auto& b : pat.bonds) {
    EnergyPatternInfo::Bond eb;
    locate(b.comp_flat_a, eb.mol1, eb.comp1);
    locate(b.comp_flat_b, eb.mol2, eb.comp2);
    if (eb.mol1 >= 0 && eb.mol2 >= 0)
      ep.bonds.push_back(eb);
  }
  return ep;
}

// Build one PatternComponent with resolved type/state indices.
PatternComponent make_energy_component(const Model& model, int type_index, const std::string& cname,
                                       BondConstraint bc, int bond_label) {
  PatternComponent pc;
  pc.name = cname;
  pc.comp_type_index =
      (type_index >= 0) ? model.molecule_types[type_index].comp_index_by_name(cname) : -1;
  pc.bond_constraint = bc;
  pc.bond_label = bond_label;
  return pc;
}

// Append this variant's context components (bound-to-anything or free) for
// reactant `reactant_idx` onto `mol`.  Binding and unbinding builds add the
// same context on both reactant and product sides so the reactant→product
// map stays the identity.
void add_energy_context(const Model& model, int type_index, int reactant_idx,
                        const std::vector<EnergyContextConstraint>& constraints,
                        PatternMolecule& mol) {
  for (const auto& cc : constraints) {
    if (cc.reactant_idx != reactant_idx)
      continue;
    mol.components.push_back(
        make_energy_component(model, type_index, cc.comp_name,
                              cc.must_be_bound ? BondConstraint::Bound : BondConstraint::Free, -1));
  }
}

// Synthesize the forward (binding) conventional rule for one expanded
// variant: molType1(site1) + molType2(site2) -> molType1(site1!1).molType2(site2!1)
// with the variant's context constraints and pre-computed rate.
Rule build_energy_binding_rule(const Model& model, const std::string& name,
                               const std::string& t1_name, const std::string& site1,
                               const std::string& t2_name, const std::string& site2,
                               const EnergyExpandedVariant& v, const std::string& rate_expr) {
  Rule r;
  r.name = name;
  r.id = name;
  int const t1 = model.mol_type_index(t1_name);
  int const t2 = model.mol_type_index(t2_name);

  // Reactant side: two separate single-molecule patterns, binding site free.
  PatternMolecule m0;
  m0.type_name = t1_name;
  m0.type_index = t1;
  m0.components.push_back(make_energy_component(model, t1, site1, BondConstraint::Free, -1));
  add_energy_context(model, t1, 0, v.constraints, m0);
  int const site2_flat = static_cast<int>(m0.components.size()); // first comp of m1
  PatternMolecule m1;
  m1.type_name = t2_name;
  m1.type_index = t2;
  m1.components.push_back(make_energy_component(model, t2, site2, BondConstraint::Free, -1));
  add_energy_context(model, t2, 1, v.constraints, m1);
  r.reactant_pattern.molecules.push_back(std::move(m0));
  r.reactant_pattern.molecules.push_back(std::move(m1));
  r.reactant_pattern_starts = {0, 1};
  r.molecularity = 2;

  // Product side: one complex, the two molecules bonded.
  PatternMolecule pm0;
  pm0.type_name = t1_name;
  pm0.type_index = t1;
  pm0.components.push_back(make_energy_component(model, t1, site1, BondConstraint::BoundTo, 0));
  add_energy_context(model, t1, 0, v.constraints, pm0);
  PatternMolecule pm1;
  pm1.type_name = t2_name;
  pm1.type_index = t2;
  pm1.components.push_back(make_energy_component(model, t2, site2, BondConstraint::BoundTo, 0));
  add_energy_context(model, t2, 1, v.constraints, pm1);
  r.product_pattern.molecules.push_back(std::move(pm0));
  r.product_pattern.molecules.push_back(std::move(pm1));
  PatternBond pb;
  pb.comp_flat_a = 0;          // site1 on product molecule 0
  pb.comp_flat_b = site2_flat; // site2 on product molecule 1
  r.product_pattern.bonds.push_back(pb);
  r.product_pattern_starts = {0};
  r.n_product_patterns = 1;

  RuleOp op;
  op.type = OpType::AddBond;
  op.comp_flat_a = 0;
  op.comp_flat_b = site2_flat;
  r.operations.push_back(op);

  // Identity reactant→product map (layouts match component-for-component).
  int const n = r.reactant_pattern.flat_comp_count();
  r.reactant_to_product_map.assign(n, -1);
  for (int i = 0; i < n; ++i)
    r.reactant_to_product_map[i] = i;

  r.same_components = (t1_name == t2_name && site1 == site2);
  r.rate_law.type = RateLawType::Ele;
  r.rate_law.rate_value = v.fwd_rate;
  r.rate_law.rate_expr = rate_expr; // re-resolved on set_param of an energy param
  return r;
}

// Synthesize the reverse (unbinding) conventional rule for one expanded
// variant: molType1(site1!1).molType2(site2!1) -> molType1(site1) + molType2(site2).
Rule build_energy_unbinding_rule(const Model& model, const std::string& name,
                                 const std::string& t1_name, const std::string& site1,
                                 const std::string& t2_name, const std::string& site2,
                                 const EnergyExpandedVariant& v, const std::string& rate_expr) {
  Rule r;
  r.name = name;
  r.id = name;
  int const t1 = model.mol_type_index(t1_name);
  int const t2 = model.mol_type_index(t2_name);

  // Reactant side: one complex with the two molecules bonded.
  PatternMolecule m0;
  m0.type_name = t1_name;
  m0.type_index = t1;
  m0.components.push_back(make_energy_component(model, t1, site1, BondConstraint::BoundTo, 0));
  add_energy_context(model, t1, 0, v.constraints, m0);
  int const site2_flat = static_cast<int>(m0.components.size());
  PatternMolecule m1;
  m1.type_name = t2_name;
  m1.type_index = t2;
  m1.components.push_back(make_energy_component(model, t2, site2, BondConstraint::BoundTo, 0));
  add_energy_context(model, t2, 1, v.constraints, m1);
  r.reactant_pattern.molecules.push_back(std::move(m0));
  r.reactant_pattern.molecules.push_back(std::move(m1));
  PatternBond pb;
  pb.comp_flat_a = 0;
  pb.comp_flat_b = site2_flat;
  r.reactant_pattern.bonds.push_back(pb);
  r.reactant_pattern_starts = {0};
  r.molecularity = 1;

  // Product side: two separate single-molecule patterns, binding site free.
  PatternMolecule pm0;
  pm0.type_name = t1_name;
  pm0.type_index = t1;
  pm0.components.push_back(make_energy_component(model, t1, site1, BondConstraint::Free, -1));
  add_energy_context(model, t1, 0, v.constraints, pm0);
  PatternMolecule pm1;
  pm1.type_name = t2_name;
  pm1.type_index = t2;
  pm1.components.push_back(make_energy_component(model, t2, site2, BondConstraint::Free, -1));
  add_energy_context(model, t2, 1, v.constraints, pm1);
  r.product_pattern.molecules.push_back(std::move(pm0));
  r.product_pattern.molecules.push_back(std::move(pm1));
  r.product_pattern_starts = {0, 1};
  r.n_product_patterns = 2;

  RuleOp op;
  op.type = OpType::DeleteBond;
  op.comp_flat_a = 0;
  op.comp_flat_b = site2_flat;
  r.operations.push_back(op);

  int const n = r.reactant_pattern.flat_comp_count();
  r.reactant_to_product_map.assign(n, -1);
  for (int i = 0; i < n; ++i)
    r.reactant_to_product_map[i] = i;

  r.same_components = (t1_name == t2_name && site1 == site2);
  r.rate_law.type = RateLawType::Ele;
  r.rate_law.rate_value = v.rev_rate;
  r.rate_law.rate_expr = rate_expr; // re-resolved on set_param of an energy param
  return r;
}

// Handle a ReactionRule that carries a `RateLaw type="Arrhenius"` (eBNGL).
// Always returns true: the Arrhenius rule is consumed here so the caller
// skips normal finalization.
//
// Each direction is processed independently — a forward (AddBond) rule emits
// the binding rules with forward rates; a reverse (DeleteBond) rule emits the
// unbinding rules with reverse rates.  For a reversible energy rule BNG2
// emits both a forward and a reverse ReactionRule, so this reconstructs the
// exact same rule set NFsim's "forward generates both" expansion does, while
// also handling a standalone unbinding rule correctly.
//
// Shapes beyond Phase 1 (matching NFsim's binding-only coverage) are NOT
// expanded: their id is appended to `unsupported_ids` so load_model raises a
// Tier-0 error, and no (broken) rule is emitted.  Refused shapes: any coupled
// operation (state change, molecule add/delete, or a mixed bond op), an
// intramolecular / ring-closure bond, a same-type homodimer (the context
// reactant attribution and molecularity-1 symmetry factor are not handled for
// automorphic reactants), and rules carrying exclude/include constraints.
bool try_expand_arrhenius(const Rule& base, const XmlNode& rl_node, const EnergyFunction& efn,
                          Model& model, bngsim::ExprTkEvaluator& ev,
                          std::unordered_map<std::string, int>& ev_ids,
                          std::vector<std::string>& unsupported_ids) {
  auto refuse = [&]() {
    unsupported_ids.push_back(base.name.empty() ? base.id : base.name);
    return true;
  };

  // The only supported shapes are a pure binding (a single AddBond) or a pure
  // unbinding (a single DeleteBond) — nothing else in the operation list.
  const RuleOp* add_bond = nullptr;
  const RuleOp* delete_bond = nullptr;
  int n_add = 0;
  int n_delete = 0;
  int n_other = 0;
  for (const auto& op : base.operations) {
    if (op.type == OpType::AddBond) {
      ++n_add;
      add_bond = &op;
    } else if (op.type == OpType::DeleteBond) {
      ++n_delete;
      delete_bond = &op;
    } else {
      ++n_other;
    }
  }
  if (n_other != 0 || n_add + n_delete != 1)
    return refuse();
  if (!base.constraints.empty())
    return refuse();

  bool const is_forward = (add_bond != nullptr);
  const RuleOp* bond_op = is_forward ? add_bond : delete_bond;
  if (bond_op->comp_flat_a < 0 || bond_op->comp_flat_b < 0)
    return refuse();

  // Resolve each bond endpoint to (molecule type, site, reactant-pattern idx).
  auto locate = [&base](int flat, std::string& mtype, std::string& site, int& rp) -> bool {
    int running = 0;
    for (int mi = 0; mi < static_cast<int>(base.reactant_pattern.molecules.size()); ++mi) {
      const auto& pm = base.reactant_pattern.molecules[mi];
      int const nc = static_cast<int>(pm.components.size());
      if (flat >= running && flat < running + nc) {
        mtype = pm.type_name;
        site = pm.components[flat - running].name;
        rp = 0;
        for (int k = 0; k < static_cast<int>(base.reactant_pattern_starts.size()); ++k)
          if (base.reactant_pattern_starts[k] <= mi)
            rp = k;
        return true;
      }
      running += nc;
    }
    return false;
  };
  std::string t1;
  std::string s1;
  std::string t2;
  std::string s2;
  int rp1 = -1;
  int rp2 = -1;
  if (!locate(bond_op->comp_flat_a, t1, s1, rp1) || !locate(bond_op->comp_flat_b, t2, s2, rp2))
    return refuse();

  if (is_forward) {
    // A binding rule joins two distinct reactant complexes.
    if (base.molecularity != 2 || rp1 == rp2)
      return refuse();
  } else if (base.molecularity != 1) {
    // An unbinding rule breaks a bond inside a single reactant complex.
    return refuse();
  }
  if (t1 == t2)
    return refuse(); // same-type homodimer — deferred

  // Arrhenius(phi, Ea0): capture both the resolved value (for the baked rate)
  // and the raw source string (for a set_param-re-resolvable symbolic rate).
  double phi = 1.0;
  double ea0 = 0.0;
  std::string phi_expr = "1";
  std::string ea0_expr = "0";
  if (auto* rc_list = find_child(rl_node, "ListOfRateConstants")) {
    int idx = 0;
    for (const auto& rcn : rc_list->children) {
      if (rcn.name != "RateConstant")
        continue;
      auto val = need_attr(rcn, "value");
      double const resolved = resolve_cached(val, ev, ev_ids);
      if (idx == 0) {
        phi = resolved;
        phi_expr = val;
      } else if (idx == 1) {
        ea0 = resolved;
        ea0_expr = val;
      }
      ++idx;
    }
  }

  // RT is symbolic when the model declares it, so overriding RT re-resolves.
  std::string const rt_expr =
      (model.parameters.count("RT") != 0) ? std::string("RT") : std::to_string(efn.rt());

  // fwd: exp(-(Ea0 + phi·ΔG)/RT);  rev: exp(-(Ea0 + (phi-1)·ΔG)/RT).
  auto rate_expr = [&](bool forward, const std::string& dg) -> std::string {
    std::string const coeff = forward ? ("(" + phi_expr + ")") : ("((" + phi_expr + ")-1)");
    return "exp(-((" + ea0_expr + ")+(" + coeff + "*(" + dg + ")))/(" + rt_expr + "))";
  };

  auto variants = efn.expand_binding(t1, s1, t2, s2, ea0, phi);
  for (std::size_t i = 0; i < variants.size(); ++i) {
    std::string const nm = base.name + "_arr_v" + std::to_string(i);
    if (is_forward)
      model.rules.push_back(build_energy_binding_rule(model, nm + "_fwd", t1, s1, t2, s2,
                                                      variants[i],
                                                      rate_expr(true, variants[i].delta_g_expr)));
    else
      model.rules.push_back(
          build_energy_unbinding_rule(model, nm + "_rev", t1, s1, t2, s2, variants[i],
                                      rate_expr(false, variants[i].delta_g_expr)));
  }
  return true;
}

// Parse TFUN CSV values
std::vector<double> parse_csv_doubles(const std::string& s, const std::string& ctx) {
  std::vector<double> vals;
  std::istringstream iss(s);
  std::string tok;
  while (std::getline(iss, tok, ',')) {
    tok = trim(tok);
    if (tok.empty())
      throw std::runtime_error(ctx + ": empty value in CSV");
    vals.push_back(std::stod(tok));
  }
  return vals;
}

TfunMethod parse_tfun_method(const std::string& s) {
  auto lo = to_lower(trim(s.empty() ? "linear" : s));
  if (lo == "linear")
    return TfunMethod::Linear;
  if (lo == "step")
    return TfunMethod::Step;
  throw std::runtime_error("Unknown TFUN method '" + s + "'");
}

// Resolve TFUN counter source
TfunCounterSource resolve_tfun_counter(const std::string& ctr_name,
                                       const std::unordered_map<std::string, double>& params,
                                       const std::unordered_set<std::string>& obs_names,
                                       const std::unordered_map<std::string, int>& func_index) {
  if (ctr_name == "time" || ctr_name == "t" || ctr_name == "time()" || ctr_name == "t()")
    return TfunCounterSource::Time;
  if (params.count(ctr_name))
    return TfunCounterSource::Parameter;
  if (obs_names.count(ctr_name))
    return TfunCounterSource::Observable;
  if (func_index.count(ctr_name))
    return TfunCounterSource::Function;
  throw std::runtime_error("TFUN counter '" + ctr_name +
                           "' does not resolve to time, parameter, observable, or function");
}

// Main model loading function
Model load_model(const std::string& xml_path,
                 std::vector<UnsupportedFeature>* unsupported_out = nullptr) {
  // Read XML file
  std::ifstream in(xml_path);
  if (!in.is_open())
    throw std::runtime_error("Cannot open XML file '" + xml_path + "'");
  std::string xml_text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  in.close();

  XmlParser parser(std::move(xml_text));
  XmlNode const root = parser.parse_document();

  // Navigate to <model>
  const XmlNode* model_node = nullptr;
  if (root.name == "model") {
    model_node = &root;
  } else {
    // SBML wrapper: <sbml><model>...</model></sbml>
    model_node = find_child(root, "model");
    if (!model_node)
      throw std::runtime_error("XML: cannot find <model> element");
  }

  Model model;
  model.xml_path = xml_path;
  auto xml_dir = std::filesystem::path(xml_path).parent_path();

  // Load-time ExprTk evaluator.  Used for the parameter cascade below
  // and, further down, for Ele/MM rate constants and initial-species
  // concentrations.  Its variables are bound by address into
  // model.parameters; that map is fully populated (every parameter
  // seeded to 0) before any expression is compiled, so no binding goes
  // stale.  This evaluator is LOCAL to load_model: the returned Model
  // is move-constructed by the caller, which relocates model.parameters,
  // so the simulator's Impl builds its own param_eval_ afterwards (see
  // Impl::build_param_evaluator).
  bngsim::ExprTkEvaluator load_eval;
  std::unordered_map<std::string, int> load_eval_ids;

  // ---- 1. Parameters ----
  auto* param_list = find_child(*model_node, "ListOfParameters");
  if (param_list) {
    // Phase 1: register every parameter (value seeded to 0) so the
    // evaluator has all variables bound before any expression compiles.
    for (auto& pn : param_list->children) {
      if (pn.name != "Parameter")
        continue;
      auto id = need_attr(pn, "id");
      auto val_str = need_attr(pn, "value");
      if (model.parameters.find(id) == model.parameters.end()) {
        model.parameters[id] = 0.0;
        model.parameter_names_ordered.push_back(id);
        load_eval.define_variable(id, &model.parameters[id]);
      }
      model.parameter_value_attrs[id] = val_str;
      // The symbolic twin, kept for the set_param cascade only (issue #23).
      // Load-time resolution below stays on `value` for NFsim parity.
      auto expr_str = opt_attr(pn, "expr");
      if (!expr_str.empty())
        model.parameter_expr_attrs[id] = expr_str;
    }
    // Phase 2: iterate to fixed point for forward references and chained
    // derivations.  BNG2 emits parameters in dependency order so a
    // single retry pass usually settles, but arbitrary XML may declare
    // `P3 = 2*P2; P2 = P1; P1 = 1` in that order — a single retry
    // resolves P2 and P1 but leaves P3 stale.  Iterate until either
    // every value is stable or the cap is hit (cap > parameter count
    // means the dependency graph has a cycle, which we can't resolve).
    const int kMaxResolvePasses = static_cast<int>(model.parameter_names_ordered.size()) + 4;
    bool hit_cap = false;
    for (int pass = 0; pass < kMaxResolvePasses; ++pass) {
      bool changed = false;
      for (auto& name : model.parameter_names_ordered) {
        const auto& val_str = model.parameter_value_attrs[name];
        try {
          double const resolved = resolve_cached(val_str, load_eval, load_eval_ids);
          if (resolved != model.parameters[name]) {
            model.parameters[name] = resolved;
            changed = true;
          }
        } catch (...) { // NOLINT(bugprone-empty-catch)
          // leave as-is — still a forward reference
        }
      }
      if (!changed) {
        break;
      }
      if (pass == kMaxResolvePasses - 1) {
        hit_cap = true;
      }
    }
    if (hit_cap) {
      std::fprintf(stderr,
                   "Warning: parameter resolution did not converge after %d passes "
                   "(parameter count = %zu); a dependency cycle is likely. "
                   "Stale parameter values may be used.\n",
                   kMaxResolvePasses, model.parameter_names_ordered.size());
    }
    // Final pass: warn on parameters whose expression still fails to
    // resolve (originally swallowed by `val = 0.0`).  Distinguishes
    // unresolved from genuinely-zero by re-evaluating against the
    // settled parameter map and checking only for thrown exceptions —
    // a parameter with expression "0" or that legitimately evaluates
    // to 0 will not throw.
    for (auto& name : model.parameter_names_ordered) {
      const auto& val_str = model.parameter_value_attrs[name];
      try {
        (void)resolve_cached(val_str, load_eval, load_eval_ids);
      } catch (const std::exception& e) {
        std::fprintf(stderr,
                     "Warning: parameter '%s' could not be resolved (expression "
                     "'%s'): %s. Using fallback value %.17g.\n",
                     name.c_str(), val_str.c_str(), e.what(), model.parameters[name]);
      } catch (...) {
        std::fprintf(stderr,
                     "Warning: parameter '%s' could not be resolved (expression "
                     "'%s'). Using fallback value %.17g.\n",
                     name.c_str(), val_str.c_str(), model.parameters[name]);
      }
    }
  }

  // ---- 2. MoleculeTypes ----
  auto* mt_list = find_child(*model_node, "ListOfMoleculeTypes");
  if (mt_list) {
    for (auto& mtn : mt_list->children) {
      if (mtn.name != "MoleculeType")
        continue;
      MoleculeType mt;
      mt.id = need_attr(mtn, "id");
      mt.name = mt.id;

      auto* ct_list = find_child(mtn, "ListOfComponentTypes");
      if (ct_list) {
        for (auto& ctn : ct_list->children) {
          if (ctn.name != "ComponentType")
            continue;
          MoleculeTypeComponent comp;
          comp.name = need_attr(ctn, "id");
          auto* states = find_child(ctn, "ListOfAllowedStates");
          if (states) {
            for (auto& sn : states->children) {
              if (sn.name != "AllowedState")
                continue;
              auto sid = need_attr(sn, "id");
              // PLUS/MINUS are pseudo-states for increment/decrement operations
              if (sid == "PLUS" || sid == "MINUS")
                continue;
              comp.allowed_states.push_back(sid);
            }
            // Sort states so that index-based increment/decrement follows the
            // natural ordering.  Numeric states sort by value; others by string.
            std::sort(comp.allowed_states.begin(), comp.allowed_states.end(),
                      [](const std::string& a, const std::string& b) {
                        // Try numeric comparison first.  clang-analyzer's
                        // path-sensitive checker occasionally flags
                        // `a.c_str()` / `b.c_str()` here as
                        // use-after-move via std::sort's swap path,
                        // but the comparator only reads the operands.
                        // `strtol`'s endptr parameter is `char**`, so these
                        // cannot be `const char*` however read-only they are
                        // here — misc-const-correctness does not model the
                        // out-param's type requirement.
                        char* ea = nullptr; // NOLINT(misc-const-correctness)
                        char* eb = nullptr; // NOLINT(misc-const-correctness)
                        // NOLINTBEGIN(clang-analyzer-cplusplus.Move)
                        long const va = std::strtol(a.c_str(), &ea, 10);
                        long const vb = std::strtol(b.c_str(), &eb, 10);
                        // NOLINTEND(clang-analyzer-cplusplus.Move)
                        bool const a_num = (ea != a.c_str() && *ea == '\0');
                        bool const b_num = (eb != b.c_str() && *eb == '\0');
                        if (a_num && b_num)
                          return va < vb;
                        if (a_num)
                          return true; // numbers before non-numbers
                        if (b_num)
                          return false;
                        return a < b;
                      });
          }
          mt.components.push_back(std::move(comp));
        }
      }

      model.molecule_type_index[mt.name] = static_cast<int>(model.molecule_types.size());
      model.molecule_types.push_back(std::move(mt));
    }
  }

  // ---- 3. Species (seed) ----
  auto* sp_list = find_child(*model_node, "ListOfSpecies");
  if (sp_list) {
    for (auto& spn : sp_list->children) {
      if (spn.name != "Species")
        continue;
      SpeciesInit si;
      si.id = need_attr(spn, "id");
      si.name = opt_attr(spn, "name");
      auto conc_str = opt_attr(spn, "concentration");
      if (conc_str.empty())
        conc_str = opt_attr(spn, "count");
      si.concentration_expr = conc_str;
      si.concentration = resolve_cached(conc_str, load_eval, load_eval_ids);

      // Map XML IDs to (mol_idx, comp_idx) within this species
      std::unordered_map<std::string, std::pair<int, int>> id_map;

      auto* mol_list = find_child(spn, "ListOfMolecules");
      if (mol_list) {
        for (auto& mn : mol_list->children) {
          if (mn.name != "Molecule")
            continue;
          SpeciesInitMol sim;
          sim.type_name = need_attr(mn, "name");
          sim.type_index = model.mol_type_index(sim.type_name);

          auto mol_xml_id = need_attr(mn, "id");
          int const mol_idx = static_cast<int>(si.molecules.size());
          id_map[mol_xml_id] = {mol_idx, -1};

          auto* cl = find_child(mn, "ListOfComponents");
          if (cl) {
            for (auto& cn : cl->children) {
              if (cn.name != "Component")
                continue;
              auto cname = need_attr(cn, "name");
              auto cstate = opt_attr(cn, "state");
              auto comp_xml_id = opt_attr(cn, "id");

              int const comp_idx = static_cast<int>(sim.comp_states.size());
              if (!comp_xml_id.empty())
                id_map[comp_xml_id] = {mol_idx, comp_idx};

              sim.comp_states.emplace_back(cname, cstate);
            }
          }
          si.molecules.push_back(std::move(sim));
        }
      }

      // Bonds in species.  Sometimes nested under <ListOfMolecules>
      // (older BNG2 emit shape).  Guard the fallback against a missing
      // <ListOfMolecules>: a degenerate <Species> without one would
      // null-deref `*mol_list`.  BNG2 doesn't emit such species, but
      // hand-crafted XML might.
      auto* bl = find_child(spn, "ListOfBonds");
      if (!bl && mol_list)
        bl = find_child(*mol_list, "ListOfBonds");
      if (bl) {
        for (auto& bn : bl->children) {
          if (bn.name != "Bond")
            continue;
          auto s1 = need_attr(bn, "site1");
          auto s2 = need_attr(bn, "site2");
          auto it1 = id_map.find(s1);
          auto it2 = id_map.find(s2);
          if (it1 != id_map.end() && it2 != id_map.end()) {
            SpeciesInitBond bond;
            bond.mol_a = it1->second.first;
            bond.comp_a = it1->second.second;
            bond.mol_b = it2->second.first;
            bond.comp_b = it2->second.second;
            si.bonds.push_back(bond);
          }
        }
      }

      // Fixed="1" attribute: build a FixedSpecies descriptor for the
      // engine's replenish_fixed_species path.  Currently-implemented
      // scope (see FixedSpecies comment in model.hpp): single
      // molecule, no bonds, at most one Fixed per MoleculeType.  Scope violations are
      // surfaced separately by scan_unsupported() as Error-level
      // warnings; we silently skip building the descriptor here so
      // --ignore-unsupported can degrade to a no-replenish fallback
      // (loud wrong rather than Tier-0 refused).
      int const init_idx = static_cast<int>(model.initial_species.size());
      bool const is_fixed = (opt_attr(spn, "Fixed") == "1");
      if (is_fixed && si.molecules.size() == 1 && si.bonds.empty()) {
        int const mt_idx = si.molecules[0].type_index;
        bool duplicate = false;
        for (auto& existing : model.fixed_species) {
          if (existing.mol_type_idx == mt_idx) {
            duplicate = true;
            break;
          }
        }
        if (!duplicate && mt_idx >= 0) {
          FixedSpecies fs;
          fs.source_init_idx = init_idx;
          fs.mol_type_idx = mt_idx;
          fs.target_count = static_cast<int>(si.concentration);
          const auto& mtype = model.molecule_types[mt_idx];
          fs.required_comp_state.assign(mtype.components.size(), -1);
          // Resolve component states declared in the seed pattern into
          // the MoleculeType's canonical component order.
          auto cmap_fs = [&]() {
            std::vector<int> mapping(si.molecules[0].comp_states.size(), -1);
            std::vector<bool> used(mtype.components.size(), false);
            for (size_t i = 0; i < si.molecules[0].comp_states.size(); ++i) {
              const auto& cname = si.molecules[0].comp_states[i].first;
              for (int j = 0; j < static_cast<int>(mtype.components.size()); ++j) {
                if (used[j])
                  continue;
                if (mtype.components[j].name == cname) {
                  mapping[i] = j;
                  used[j] = true;
                  break;
                }
              }
            }
            return mapping;
          }();
          for (size_t i = 0; i < si.molecules[0].comp_states.size(); ++i) {
            const auto& [cname, cstate] = si.molecules[0].comp_states[i];
            int const actual_ci = cmap_fs[i];
            if (actual_ci < 0 || cstate.empty())
              continue;
            int const sidx = mtype.state_index(actual_ci, cstate);
            if (sidx >= 0)
              fs.required_comp_state[actual_ci] = sidx;
          }
          model.fixed_species.push_back(std::move(fs));
        }
      }

      model.initial_species.push_back(std::move(si));
    }
  }

  // ---- 4. Observables (must be before rules) ----
  std::unordered_set<std::string> obs_name_set;
  auto* obs_list = find_child(*model_node, "ListOfObservables");
  if (obs_list) {
    for (auto& on : obs_list->children) {
      if (on.name != "Observable")
        continue;
      Observable obs;
      obs.id = need_attr(on, "id");
      obs.name = need_attr(on, "name");
      obs.type = need_attr(on, "type");

      auto* pat_list = find_child(on, "ListOfPatterns");
      if (pat_list) {
        for (auto& pn : pat_list->children) {
          if (pn.name != "Pattern")
            continue;
          std::unordered_map<std::string, std::pair<int, int>> id_flat;
          auto pat = parse_pattern(pn, model, &id_flat);

          // Species observable quantifier
          auto rel = opt_attr(pn, "relation");
          auto qty = opt_attr(pn, "quantity");
          if (!rel.empty())
            pat.relation = rel;
          if (!qty.empty())
            pat.quantity = std::stoi(qty);

          obs.patterns.push_back(std::move(pat));
        }
      }

      obs_name_set.insert(obs.name);
      model.observable_names_ordered.push_back(obs.name);
      model.observable_index[obs.name] = static_cast<int>(model.observables.size());
      model.observables.push_back(std::move(obs));
    }
  }

  // ---- 5. Functions ----
  // Observables a local function references both tagged and bare — RM
  // resolves them as tagged and warns (see the split below).
  std::vector<std::string> mixed_scope_observables;
  auto* func_list = find_child(*model_node, "ListOfFunctions");
  if (func_list) {
    for (auto& fn : func_list->children) {
      if (fn.name != "Function")
        continue;
      GlobalFunction gf;
      gf.name = need_attr(fn, "id");

      auto type_attr = opt_attr(fn, "type");

      // Parse arguments (local function support)
      auto* arg_list_fn = find_child(fn, "ListOfArguments");
      if (arg_list_fn) {
        for (auto& an : arg_list_fn->children) {
          if (an.name != "Argument")
            continue;
          gf.argument_names.push_back(need_attr(an, "id"));
        }
      }

      // Collect the observables this function references.  Which of them
      // are evaluated locally is decided further down, once the
      // expression text is in hand — `<ListOfReferences>` cannot answer it
      // (issue #38).
      std::vector<std::string> referenced_observables;
      auto* ref_list = find_child(fn, "ListOfReferences");
      if (ref_list) {
        for (auto& rn : ref_list->children) {
          if (rn.name != "Reference")
            continue;
          auto rtype = opt_attr(rn, "type");
          if (rtype == "Observable") {
            referenced_observables.push_back(need_attr(rn, "name"));
          }
        }
      }

      if (to_lower(type_attr) == "tfun") {
        // TFUN function
        gf.is_tfun = true;
        gf.tfun_counter_name = trim(opt_attr(fn, "ctrName"));
        auto method = parse_tfun_method(opt_attr(fn, "method"));

        auto file_attr = trim(opt_attr(fn, "file"));
        auto x_attr = opt_attr(fn, "xData");
        auto y_attr = opt_attr(fn, "yData");

        if (!file_attr.empty()) {
          // Search order for the .tfun data file, in priority order:
          //   1. <xml_dir>/<file>           — typical: BNG2 ran in-place on the BNGL,
          //                                    leaving XML and .tfun side by side.
          //   2. <xml_dir>/../<file>        — typical for our test harness: XML is
          //                                    generated to tests/.../xml/ but the
          //                                    .tfun lives next to the BNGL one
          //                                    directory up.
          // (Absolute file_attr paths bypass both searches via filesystem rules.)
          auto candidate1 = (xml_dir / file_attr).lexically_normal();
          auto candidate2 = (xml_dir / ".." / file_attr).lexically_normal();
          std::string tfun_path;
          std::error_code ec;
          // The two assignment targets look textually similar but
          // resolve to distinct paths (xml_dir/<file> vs xml_dir/../<file>);
          // the chain isn't a clone.
          // NOLINTNEXTLINE(bugprone-branch-clone)
          if (std::filesystem::exists(candidate1, ec)) {
            tfun_path = candidate1.string();
          } else if (std::filesystem::exists(candidate2, ec)) {
            tfun_path = candidate2.string();
          } else {
            tfun_path = candidate1.string(); // let from_file() raise the canonical error
          }
          gf.tfun = std::make_shared<TableFunction>(
              TableFunction::from_file(gf.name, tfun_path, gf.tfun_counter_name, method));
        } else {
          auto xs = parse_csv_doubles(x_attr, gf.name);
          auto ys = parse_csv_doubles(y_attr, gf.name);
          gf.tfun = std::make_shared<TableFunction>(gf.name, std::move(xs), std::move(ys),
                                                    gf.tfun_counter_name, method);
        }

        // Capture the wrapper expression (may contain __TFUN__VAL__).
        // Actual compilation happens at engine init (ExprTk); a pure-TFUN
        // function whose expression won't compile falls back to the raw
        // table value there.
        auto* expr_node = find_child(fn, "Expression");
        if (expr_node && !expr_node->text.empty()) {
          gf.expression_text = trim(expr_node->text);
          // Replace __TFUN__VAL__ with the magic __tfun_NAME__ slot name.
          auto& et = gf.expression_text;
          // Two BNG2 emit conventions for the lookup-result sentinel:
          //   - BNG2 2.9.3 (uppercase TFUN()):   "__TFUN__VAL__" (13 chars)
          //   - BNG2 dev / fix-tfun-has-tfuns-reset (lowercase tfun()):
          //                                      "__TFUN_VAL__"  (12 chars)
          // Try the longer form first so we don't half-substitute it.
          for (const auto& [sentinel, slen] :
               std::initializer_list<std::pair<const char*, std::size_t>>{
                   {"__TFUN__VAL__", 13},
                   {"__TFUN_VAL__", 12},
               }) {
            auto pos = et.find(sentinel);
            if (pos != std::string::npos) {
              et.replace(pos, slen, "__tfun_" + gf.name + "__");
              break;
            }
          }
        }
      } else {
        // Regular function — capture the expression source; the engine
        // compiles it (and surfaces any syntax error) at init time.
        auto* expr_node = find_child(fn, "Expression");
        if (expr_node && !expr_node->text.empty())
          gf.expression_text = trim(expr_node->text);
      }

      // `reactant_N()` placeholder (see GlobalFunction::reactant_count_index).
      // Recognized by shape alone, since that is all BNG2 emits: the name
      // `reactant_` plus a single digit 1-9, no arguments, and no expression
      // body.  A function that has a body of its own is an ordinary function
      // whatever it is called, so the model keeps whatever it wrote.
      if (gf.expression_text.empty() && gf.argument_names.empty() && !gf.is_tfun &&
          gf.name.size() == 10 && gf.name.compare(0, 9, "reactant_") == 0 && gf.name[9] >= '1' &&
          gf.name[9] <= '9')
        gf.reactant_count_index = gf.name[9] - '0';

      // Split the referenced observables by scope (issue #38).  An
      // observable is local iff the expression applies it to one of THIS
      // function's own arguments; a bare reference is the global,
      // system-wide count and must keep reading its global value.  BNG2
      // states the same split in the network it generates for the model,
      // folding the bare observable into the rate expression and
      // resolving only the tagged one per instance:
      //
      //     1 _R_local1() ((kc*Obs_Src)*1)
      //
      // Before this split RM localized every referenced observable, so a
      // bare one evaluated at the tagged molecule, read 0 whenever its
      // pattern was absent from that molecule's complex — which for a
      // system-wide quantity is essentially always — and zeroed the
      // propensity.  The rule then never fired, silently.
      if (gf.is_local()) {
        auto const scoped =
            expr::classify_by_tag_application(gf.expression_text, gf.argument_names);
        auto const& tagged = scoped.tag_applied;
        for (auto& obs_name : referenced_observables) {
          if (std::find(tagged.begin(), tagged.end(), obs_name) == tagged.end()) {
            gf.global_observable_names.push_back(obs_name);
            continue;
          }
          gf.local_observable_names.push_back(obs_name);
          // Written BOTH ways in one function (`O + O(x)`) the observable
          // wants two values out of one eval-layout slot, which RM cannot
          // express.  The tagged reading wins — it is the one that needs
          // the local machinery — and the model is flagged rather than
          // quietly mis-evaluated.
          if (std::find(scoped.bare.begin(), scoped.bare.end(), obs_name) != scoped.bare.end())
            mixed_scope_observables.push_back("'" + obs_name + "' in function '" + gf.name + "'");
        }
      }

      model.function_index[gf.name] = static_cast<int>(model.functions.size());
      model.functions.push_back(std::move(gf));
    }
  }

  // Resolve TFUN counter sources
  for (auto& gf : model.functions) {
    if (!gf.is_tfun)
      continue;
    gf.tfun_counter_source = resolve_tfun_counter(gf.tfun_counter_name, model.parameters,
                                                  obs_name_set, model.function_index);
  }

  // ---- 5b. EnergyPatterns (eBNGL) ----
  // Parse <ListOfEnergyPatterns> into an EnergyFunction so that any
  // `Arrhenius` energy rules can be expanded into conventional rules in the
  // ReactionRules loop below (Sekar load-time expansion — see
  // energy_expand.hpp).  A bare energy-pattern block paired with
  // Function-type rate laws (which inline the Boltzmann factors) needs no
  // expansion and is unaffected: the function is only consulted when a rule
  // declares RateLaw type="Arrhenius".
  double energy_rt = 2.478; // NFsim default: R·T ≈ 2.478 kJ/mol at 298 K
  if (auto rt_it = model.parameters.find("RT"); rt_it != model.parameters.end())
    energy_rt = rt_it->second;
  EnergyFunction energy_fn(energy_rt);
  // Arrhenius energy rules whose shape RM does not expand (see
  // try_expand_arrhenius); surfaced as Tier-0 errors below.
  std::vector<std::string> unsupported_arrhenius_ids;
  if (auto* ep_list = find_child(*model_node, "ListOfEnergyPatterns")) {
    for (auto& epn : ep_list->children) {
      if (epn.name != "EnergyPattern")
        continue;
      auto ep_id = opt_attr(epn, "id");
      auto ep_expr = opt_attr(epn, "expression");
      double const energy = resolve_cached(ep_expr, load_eval, load_eval_ids);
      // The graph lives under a <Pattern> child (BNG 2.9.3) or, in older
      // emit conventions, directly as <ListOfMolecules>/<ListOfBonds>.
      auto* pat_node = find_child(epn, "Pattern");
      std::unordered_map<std::string, std::pair<int, int>> ep_id_map;
      const Pattern pat = parse_pattern(pat_node != nullptr ? *pat_node : epn, model, &ep_id_map);
      energy_fn.add_pattern(pattern_to_energy_info(ep_id, energy, ep_expr, pat));
    }
  }

  // ---- 6. ReactionRules ----
  auto* rr_list = find_child(*model_node, "ListOfReactionRules");
  if (rr_list) {
    for (auto& rrn : rr_list->children) {
      if (rrn.name != "ReactionRule")
        continue;
      Rule rule;
      rule.id = need_attr(rrn, "id");
      rule.name = opt_attr(rrn, "name");
      if (rule.name.empty())
        rule.name = rule.id;

      auto sym = opt_attr(rrn, "symmetry_factor");
      if (!sym.empty())
        rule.symmetry_factor = std::stod(sym);

      // ID map for component resolution across patterns
      std::unordered_map<std::string, std::pair<int, int>> reactant_id_map;
      std::unordered_set<std::string> reactant_pattern_ids; // IDs that are ReactantPattern-level
      std::vector<std::string> rp_id_list; // ReactantPattern IDs in order (for constraint matching)

      // Reactant patterns
      auto* rp_list = find_child(rrn, "ListOfReactantPatterns");
      int rp_count = 0;
      if (rp_list) {
        for (auto& rpn : rp_list->children) {
          if (rpn.name != "ReactantPattern")
            continue;
          int const mol_offset = static_cast<int>(rule.reactant_pattern.molecules.size());
          rule.reactant_pattern_starts.push_back(mol_offset);

          // Register ReactantPattern ID → first molecule in this pattern
          auto rp_id = opt_attr(rpn, "id");
          if (!rp_id.empty()) {
            reactant_id_map[rp_id] = {mol_offset, -1};
            reactant_pattern_ids.insert(rp_id);
            rp_id_list.push_back(rp_id);
          }

          // Parse sub-pattern; its mol indices start from 0
          std::unordered_map<std::string, std::pair<int, int>> sub_id_map;
          auto sub = parse_pattern(rpn, model, &sub_id_map);

          // Merge sub_id_map into reactant_id_map with offset
          for (auto& [id, pos] : sub_id_map)
            reactant_id_map[id] = {pos.first + mol_offset, pos.second};

          // Also adjust bond flat indices
          int const comp_offset = rule.reactant_pattern.flat_comp_count();
          for (auto& mol : sub.molecules)
            rule.reactant_pattern.molecules.push_back(std::move(mol));
          for (auto& b : sub.bonds) {
            b.comp_flat_a += comp_offset;
            b.comp_flat_b += comp_offset;
            rule.reactant_pattern.bonds.push_back(b);
          }

          ++rp_count;
        }
      }
      rule.molecularity = rp_count;

      // Product patterns
      std::unordered_map<std::string, std::pair<int, int>> product_id_map;
      std::vector<std::string> pp_id_list; // ProductPattern IDs in order
      auto* pp_list = find_child(rrn, "ListOfProductPatterns");
      rule.n_product_patterns = 0;
      if (pp_list) {
        for (auto& ppn : pp_list->children) {
          if (ppn.name != "ProductPattern")
            continue;
          auto pp_id = opt_attr(ppn, "id");
          if (!pp_id.empty())
            pp_id_list.push_back(pp_id);
          ++rule.n_product_patterns;
          int const mol_offset = static_cast<int>(rule.product_pattern.molecules.size());
          rule.product_pattern_starts.push_back(mol_offset);
          std::unordered_map<std::string, std::pair<int, int>> sub_id_map;
          auto sub = parse_pattern(ppn, model, &sub_id_map);
          for (auto& [id, pos] : sub_id_map)
            product_id_map[id] = {pos.first + mol_offset, pos.second};
          int const comp_offset = rule.product_pattern.flat_comp_count();
          for (auto& mol : sub.molecules)
            rule.product_pattern.molecules.push_back(std::move(mol));
          for (auto& b : sub.bonds) {
            b.comp_flat_a += comp_offset;
            b.comp_flat_b += comp_offset;
            rule.product_pattern.bonds.push_back(b);
          }
        }
      }

      // Map: reactant -> product component mapping
      auto* map_node = find_child(rrn, "Map");
      int const n_rcomp = rule.reactant_pattern.flat_comp_count();
      rule.reactant_to_product_map.assign(n_rcomp, -1);
      if (map_node) {
        for (auto& mi : map_node->children) {
          if (mi.name != "MapItem")
            continue;
          auto src = need_attr(mi, "sourceID");
          auto tgt = opt_attr(mi, "targetID");
          if (tgt.empty())
            continue; // unmapped (deleted) component
          auto sit = reactant_id_map.find(src);
          auto tit = product_id_map.find(tgt);
          if (sit != reactant_id_map.end() && tit != product_id_map.end()) {
            if (sit->second.second >= 0 && tit->second.second >= 0) {
              int const src_flat =
                  rule.reactant_pattern.flat_index(sit->second.first, sit->second.second);
              int const tgt_flat =
                  rule.product_pattern.flat_index(tit->second.first, tit->second.second);
              if (src_flat >= 0 && src_flat < n_rcomp)
                rule.reactant_to_product_map[src_flat] = tgt_flat;
            }
          }
        }
      }

      // Operations
      auto* ops_node = find_child(rrn, "ListOfOperations");
      if (ops_node) {
        for (auto& opn : ops_node->children) {
          RuleOp op;

          if (opn.name == "StateChange") {
            op.type = OpType::StateChange;
            auto site = need_attr(opn, "site");
            auto sit = reactant_id_map.find(site);
            if (sit != reactant_id_map.end() && sit->second.second >= 0) {
              op.comp_flat =
                  rule.reactant_pattern.flat_index(sit->second.first, sit->second.second);
            }
            op.new_state = need_attr(opn, "finalState");
            if (op.new_state == "PLUS") {
              op.is_increment = true;
              op.new_state = "";
            } else if (op.new_state == "MINUS") {
              op.is_decrement = true;
              op.new_state = "";
            } else {
              // Resolve state index
              if (sit != reactant_id_map.end()) {
                auto& pm = rule.reactant_pattern.molecules[sit->second.first];
                auto& pc = pm.components[sit->second.second];
                if (pm.type_index >= 0 && pc.comp_type_index >= 0)
                  op.new_state_index = model.molecule_types[pm.type_index].state_index(
                      pc.comp_type_index, op.new_state);
              }
            }
            rule.operations.push_back(std::move(op));

          } else if (opn.name == "AddBond") {
            op.type = OpType::AddBond;
            auto s1 = need_attr(opn, "site1");
            auto s2 = need_attr(opn, "site2");
            // Try reactant pattern first, then product pattern
            auto rit1 = reactant_id_map.find(s1);
            auto rit2 = reactant_id_map.find(s2);
            if (rit1 != reactant_id_map.end() && rit1->second.second >= 0)
              op.comp_flat_a =
                  rule.reactant_pattern.flat_index(rit1->second.first, rit1->second.second);
            if (rit2 != reactant_id_map.end() && rit2->second.second >= 0)
              op.comp_flat_b =
                  rule.reactant_pattern.flat_index(rit2->second.first, rit2->second.second);
            // If a site is on a product-pattern molecule (newly added),
            // store the product mol index and the molecule-TYPE component
            // index (not the pattern-local index) for fire-time resolution.
            if (op.comp_flat_a < 0) {
              auto pit = product_id_map.find(s1);
              if (pit != product_id_map.end()) {
                op.product_mol_a = pit->second.first;
                // Convert pattern comp index → type comp index
                auto& ppmc = rule.product_pattern.molecules[pit->second.first]
                                 .components[pit->second.second];
                op.product_comp_a = ppmc.comp_type_index;
              }
            }
            if (op.comp_flat_b < 0) {
              auto pit = product_id_map.find(s2);
              if (pit != product_id_map.end()) {
                op.product_mol_b = pit->second.first;
                auto& ppmc = rule.product_pattern.molecules[pit->second.first]
                                 .components[pit->second.second];
                op.product_comp_b = ppmc.comp_type_index;
              }
            }
            rule.operations.push_back(std::move(op));

          } else if (opn.name == "DeleteBond") {
            op.type = OpType::DeleteBond;
            auto s1 = need_attr(opn, "site1");
            auto s2 = need_attr(opn, "site2");
            auto it1 = reactant_id_map.find(s1);
            auto it2 = reactant_id_map.find(s2);
            if (it1 != reactant_id_map.end() && it1->second.second >= 0)
              op.comp_flat_a =
                  rule.reactant_pattern.flat_index(it1->second.first, it1->second.second);
            if (it2 != reactant_id_map.end() && it2->second.second >= 0)
              op.comp_flat_b =
                  rule.reactant_pattern.flat_index(it2->second.first, it2->second.second);
            // Parse ensureConnected: when "1", the bond break must leave
            // both molecules in the same complex (ring-bond-only unbinding).
            auto ec = opn.attributes.find("ensureConnected");
            if (ec != opn.attributes.end() && ec->second == "1")
              op.ensure_connected = true;
            rule.operations.push_back(std::move(op));

          } else if (opn.name == "Add") {
            op.type = OpType::AddMolecule;
            auto add_id = need_attr(opn, "id");
            // Find the molecule in product pattern
            auto pit = product_id_map.find(add_id);
            if (pit != product_id_map.end()) {
              op.add_product_mol_idx = pit->second.first;
              auto& prod_mol = rule.product_pattern.molecules[pit->second.first];
              op.add_spec.type_index = prod_mol.type_index;
              for (auto& pc : prod_mol.components) {
                if (!pc.required_state.empty() && pc.required_state_index >= 0)
                  op.add_spec.comp_states.emplace_back(pc.comp_type_index, pc.required_state_index);
              }
            }
            rule.operations.push_back(std::move(op));

          } else if (opn.name == "Delete") {
            op.type = OpType::DeleteMolecule;
            auto del_id = need_attr(opn, "id");
            auto del_mols = opt_attr(opn, "DeleteMolecules");
            // DeleteMolecules="1" means delete only the specified molecule(s),
            // NOT the entire connected species.  Absence or "0" means delete
            // the whole species.
            op.delete_connected = (del_mols != "1");
            auto dit = reactant_id_map.find(del_id);
            if (dit != reactant_id_map.end())
              op.delete_pattern_mol_idx = dit->second.first;
            rule.operations.push_back(std::move(op));
          }
        }
      }

      // Exclude/Include Reactants/Products constraints
      {
        struct ConstraintListDef {
          const char* element;
          bool is_exclude;
          bool is_product;
          const std::vector<std::string>* id_list;
        };
        ConstraintListDef cdefs[] = {
            {"ListOfExcludeReactants", true, false, &rp_id_list},
            {"ListOfIncludeReactants", false, false, &rp_id_list},
            {"ListOfExcludeProducts", true, true, &pp_id_list},
            {"ListOfIncludeProducts", false, true, &pp_id_list},
        };
        for (auto& cd : cdefs) {
          for (auto& child : rrn.children) {
            if (child.name != cd.element)
              continue;
            // The id attribute matches a ReactantPattern/ProductPattern id
            auto list_id = opt_attr(child, "id");
            int pat_idx = -1;
            for (int i = 0; i < static_cast<int>(cd.id_list->size()); ++i) {
              if ((*cd.id_list)[i] == list_id) {
                pat_idx = i;
                break;
              }
            }
            if (pat_idx < 0) {
              std::fprintf(stderr,
                           "Warning: %s id='%s' in rule '%s' does not "
                           "match any %s\n",
                           cd.element, list_id.c_str(), rule.name.c_str(),
                           cd.is_product ? "ProductPattern" : "ReactantPattern");
              continue;
            }
            // Parse each Pattern child
            for (auto& pn : child.children) {
              if (pn.name != "Pattern")
                continue;
              auto cpat = parse_pattern(pn, model);
              Rule::Constraint c;
              c.pattern_idx = pat_idx;
              c.pattern = std::move(cpat);
              c.is_exclude = cd.is_exclude;
              c.is_product = cd.is_product;
              rule.constraints.push_back(std::move(c));
            }
          }
        }
      }

      // Rate law
      auto* rl_node = find_child(rrn, "RateLaw");
      if (rl_node) {
        auto rl_type = opt_attr(*rl_node, "type");
        auto totalrate = opt_attr(*rl_node, "totalrate");
        rule.rate_law.is_total_rate = (totalrate == "1");

        if (rl_type == "Arrhenius") {
          // eBNGL energy rule: expand this direction into conventional rules
          // with pre-computed rates and push them directly, then skip the
          // normal single-rule finalization.  Unsupported shapes append their
          // id to unsupported_arrhenius_ids (surfaced as Tier-0 errors).
          try_expand_arrhenius(rule, *rl_node, energy_fn, model, load_eval, load_eval_ids,
                               unsupported_arrhenius_ids);
          continue;
        }

        // Reactant pattern the single local-function argument is tagged on,
        // -1 until resolved.  Only consulted for the DOR1 normalization below.
        int local_arg_rp_idx = -1;

        if (rl_type == "Ele") {
          rule.rate_law.type = RateLawType::Ele;
          auto* rc_list = find_child(*rl_node, "ListOfRateConstants");
          if (rc_list) {
            for (auto& rcn : rc_list->children) {
              if (rcn.name != "RateConstant")
                continue;
              auto val = need_attr(rcn, "value");
              rule.rate_law.rate_expr = val;
              rule.rate_law.rate_value = resolve_cached(val, load_eval, load_eval_ids);
              break;
            }
          }
        } else if (rl_type == "Function") {
          rule.rate_law.type = RateLawType::Function;
          rule.rate_law.is_dynamic = true;
          rule.rate_law.function_name = opt_attr(*rl_node, "name");

          // Check for arguments (local function)
          auto* arg_list = find_child(*rl_node, "ListOfArguments");
          if (arg_list && !arg_list->children.empty()) {
            rule.rate_law.is_local = true;
            // Determine if the argument is bound to a molecule or a pattern.
            // Molecule IDs (e.g. "RR7_RP1_M1") are in reactant_id_map but
            // NOT in reactant_pattern_ids. Pattern IDs (e.g. "RR9_RP1") are
            // in both. Both have comp_index == -1 so we cannot distinguish
            // them by comp_index alone.
            for (auto& an : arg_list->children) {
              auto argval = opt_attr(an, "value");
              if (!argval.empty()) {
                auto rit = reactant_id_map.find(argval);
                if (rit != reactant_id_map.end() &&
                    reactant_pattern_ids.find(argval) == reactant_pattern_ids.end()) {
                  rule.rate_law.local_arg_is_molecule = true;
                } else if (rit == reactant_id_map.end()) {
                  std::fprintf(stderr,
                               "Warning: local function arg '%s' in "
                               "rule '%s' not found in reactant_id_map; defaulting to "
                               "complex-wide scope\n",
                               argval.c_str(), rule.name.c_str());
                }
                // Which reactant pattern carries the tag?  Irrelevant for a
                // unimolecular rule (there is only one) but load-bearing for
                // the DOR1 normalization below, which has to know whether the
                // per-instance factor belongs on the A or the B sampler slot.
                if (rit != reactant_id_map.end()) {
                  int const mol_off = rit->second.first;
                  for (int k = 0; k < static_cast<int>(rule.reactant_pattern_starts.size()); ++k) {
                    if (rule.reactant_pattern_starts[k] <= mol_off)
                      local_arg_rp_idx = k;
                  }
                }
              }
            }
          }
        } else if (rl_type == "FunctionProduct") {
          // NFsim's DOR2: two per-reactant local-function factors, multiplied.
          // <RateLaw type="FunctionProduct" name1=".." name2="..">
          //   <ListOfArguments1><Argument id=".." value="<reactant>"/></..>
          //   <ListOfArguments2><Argument id=".." value="<reactant>"/></..>
          // Each factor's argument `value` points at a ReactantPattern (id in
          // reactant_pattern_ids -> complex-wide scope) or a tagged molecule
          // inside one (-> per-molecule scope), exactly like a single local
          // Function.  We map whichever factor references reactant pattern 0
          // to the A-side fields and the one referencing pattern 1 to the
          // B-side fields so the engine's fixed A/B sampler slots line up.
          rule.rate_law.type = RateLawType::FunctionProduct;
          rule.rate_law.is_dynamic = true;

          // Resolve an Argument `value` to (reactant-pattern-index, per_mol).
          auto resolve_factor = [&](const std::string& argval, int& out_rp_idx, bool& out_per_mol) {
            out_rp_idx = -1;
            out_per_mol = false;
            // Direct ReactantPattern-id match -> complex-wide scope.
            for (int k = 0; k < static_cast<int>(rp_id_list.size()); ++k) {
              if (rp_id_list[k] == argval) {
                out_rp_idx = k;
                out_per_mol = false;
                return;
              }
            }
            // Otherwise a tagged molecule id -> per-molecule scope; locate its
            // owning reactant pattern by molecule offset.
            auto it = reactant_id_map.find(argval);
            if (it != reactant_id_map.end()) {
              int const mol_off = it->second.first;
              for (int k = 0; k < static_cast<int>(rule.reactant_pattern_starts.size()); ++k) {
                if (rule.reactant_pattern_starts[k] <= mol_off)
                  out_rp_idx = k;
              }
              out_per_mol = (reactant_pattern_ids.find(argval) == reactant_pattern_ids.end());
            }
          };

          auto first_arg_value = [&](const char* list_name) -> std::string {
            auto* al = find_child(*rl_node, list_name);
            if (al) {
              for (auto& an : al->children) {
                if (an.name == "Argument")
                  return opt_attr(an, "value");
              }
            }
            return "";
          };

          struct Factor {
            std::string fn;
            int rp_idx = -1;
            bool per_mol = false;
          };
          std::array<Factor, 2> factors;
          factors[0].fn = opt_attr(*rl_node, "name1");
          factors[1].fn = opt_attr(*rl_node, "name2");
          resolve_factor(first_arg_value("ListOfArguments1"), factors[0].rp_idx,
                         factors[0].per_mol);
          resolve_factor(first_arg_value("ListOfArguments2"), factors[1].rp_idx,
                         factors[1].per_mol);

          // Assign the pattern-0 factor to the A-side and pattern-1 to the
          // B-side.  Default to declaration order if a reference can't be
          // resolved (single-reactant edge cases).
          int a_factor = 0, b_factor = 1;
          if (factors[0].rp_idx == 1 || factors[1].rp_idx == 0) {
            a_factor = 1;
            b_factor = 0;
          }
          rule.rate_law.function_name = factors[a_factor].fn;
          rule.rate_law.is_local = true;
          rule.rate_law.local_arg_is_molecule = factors[a_factor].per_mol;
          rule.rate_law.function_name_b = factors[b_factor].fn;
          rule.rate_law.is_local_b = true;
          rule.rate_law.local_arg_is_molecule_b = factors[b_factor].per_mol;
        } else if (rl_type == "MM") {
          rule.rate_law.type = RateLawType::MM;
          auto* rc_list = find_child(*rl_node, "ListOfRateConstants");
          if (rc_list) {
            int idx = 0;
            for (auto& rcn : rc_list->children) {
              if (rcn.name != "RateConstant")
                continue;
              auto val = need_attr(rcn, "value");
              if (idx == 0) {
                rule.rate_law.mm_kcat_expr = val;
                rule.rate_law.mm_kcat = resolve_cached(val, load_eval, load_eval_ids);
              } else if (idx == 1) {
                rule.rate_law.mm_Km_expr = val;
                rule.rate_law.mm_Km = resolve_cached(val, load_eval, load_eval_ids);
              }
              ++idx;
            }
          }
        }

        // NFsim DOR1 — a bimolecular rule with ONE tagged reactant, e.g.
        //   R: S(s~0) + E()%x -> S(s~1) + E()%x   lf(x)
        // The per-instance factor applies to the tagged reactant only; the
        // untagged one contributes its plain embedding count, so the rule is
        // a FunctionProduct whose other factor is the constant 1.  Normalizing
        // it here means the engine's DOR2 propensity, incremental update and
        // sampler cover it unchanged — the alternative was a fourth
        // propensity branch duplicating all three.
        //
        // Without this the rule kept RateLawType::Function with molecularity
        // 2, which no branch of recompute_rule_state handled: it fell through
        // to the mass-action path with the local function evaluated on no
        // molecule at all, leaving rs.local_propensity_total at 0 while
        // has_local_rates stayed true.  The first incremental_update then read
        // that never-populated accumulator as the propensity and the rule went
        // inert after a single firing (issue #34).
        if (rule.rate_law.type == RateLawType::Function && rule.rate_law.is_local &&
            rule.molecularity == 2) {
          rule.rate_law.type = RateLawType::FunctionProduct;
          rule.rate_law.is_local_b = true;
          if (local_arg_rp_idx == 1) {
            // Tag on reactant pattern 1 — move the factor to the B slot.
            rule.rate_law.function_name_b = rule.rate_law.function_name;
            rule.rate_law.local_arg_is_molecule_b = rule.rate_law.local_arg_is_molecule;
            rule.rate_law.function_name.clear();
            rule.rate_law.unity_factor_a = true;
            // Per-molecule scope on the unity side skips the per-complex
            // rate cache, which would only churn to memoize the constant 1.
            rule.rate_law.local_arg_is_molecule = true;
          } else {
            rule.rate_law.function_name_b.clear();
            rule.rate_law.unity_factor_b = true;
            rule.rate_law.local_arg_is_molecule_b = true;
          }
        }
      }

      // Detect same_components
      rule.same_components = false;
      if (rule.molecularity == 2) {
        for (auto& op : rule.operations) {
          if (op.type == OpType::AddBond && op.comp_flat_a >= 0 && op.comp_flat_b >= 0) {
            // Walk reactant-pattern molecules to find which one each
            // bond endpoint lives in.
            int const flat_a = op.comp_flat_a, flat_b = op.comp_flat_b;
            int mol_a = -1, local_a = -1, mol_b = -1, local_b = -1;
            int running = 0;
            for (int mi = 0; mi < static_cast<int>(rule.reactant_pattern.molecules.size()); ++mi) {
              int const nc =
                  static_cast<int>(rule.reactant_pattern.molecules[mi].components.size());
              if (flat_a >= running && flat_a < running + nc) {
                mol_a = mi;
                local_a = flat_a - running;
              }
              if (flat_b >= running && flat_b < running + nc) {
                mol_b = mi;
                local_b = flat_b - running;
              }
              running += nc;
            }

            if (mol_a >= 0 && mol_b >= 0) {
              auto& ma = rule.reactant_pattern.molecules[mol_a];
              auto& mb = rule.reactant_pattern.molecules[mol_b];
              if (ma.type_name == mb.type_name && local_a >= 0 && local_b >= 0) {
                auto& ca = ma.components[local_a];
                auto& cb = mb.components[local_b];
                if (ca.name == cb.name)
                  rule.same_components = true;
              }
            }
            break; // only check first AddBond
          }
        }
      }

      // Do NOT modify rate by symmetry_factor during parsing.
      // Instead, symmetry_factor is applied in compute_propensity:
      //   - Unimolecular: propensity = a_total * rate * sf
      //     (sf corrects for pattern automorphisms like swapping identical
      //     molecules in M(a!1).M(a!1), where a_total=2 per dimer)
      //   - Bimolecular: sf not applied (same_components formula and
      //     embedding_correction already handle the combinatorics)
      //   - MM(kcat,Km): sf scales the count of the reactant the rule
      //     transforms, *inside* the law rather than on the finished
      //     propensity, since the law is not linear in it (issue #37)

      model.rules.push_back(std::move(rule));
    }
  }

  // ---- Mark rate-dependent observables ----
  // An observable is rate-dependent only if it is transitively reachable
  // from a rate law.  Output-only functions that reference observables
  // do NOT make those observables rate-dependent.
  {
    // Build a dependency graph: function name -> set of names it references
    // (parameters, observables, or other functions).
    std::unordered_map<std::string, std::vector<std::string>> func_deps;
    for (auto& gf : model.functions) {
      std::vector<std::string> deps = expr::collect_variables(gf.expression_text);
      if (gf.is_tfun && gf.tfun_counter_source == TfunCounterSource::Observable)
        deps.push_back(gf.tfun_counter_name);
      if (gf.is_tfun && gf.tfun_counter_source == TfunCounterSource::Function)
        deps.push_back(gf.tfun_counter_name);
      func_deps[gf.name] = std::move(deps);
    }

    // Seed: collect names directly referenced by rate laws
    std::unordered_set<std::string> seeds;
    for (auto& rule : model.rules) {
      if (!rule.rate_law.function_name.empty())
        seeds.insert(rule.rate_law.function_name);
      if (!rule.rate_law.function_name_b.empty())
        seeds.insert(rule.rate_law.function_name_b);
      if (rule.rate_law.uses_tfun &&
          rule.rate_law.tfun_counter_source == TfunCounterSource::Observable)
        seeds.insert(rule.rate_law.tfun_counter_name);
      if (rule.rate_law.uses_tfun &&
          rule.rate_law.tfun_counter_source == TfunCounterSource::Function)
        seeds.insert(rule.rate_law.tfun_counter_name);
    }

    // Transitive closure: expand seeds through function dependencies.
    // For local functions, observables in their local_observable_names are
    // evaluated per-molecule (in evaluate_local_rate), not globally — so
    // they don't need global recomputation and should not be marked
    // rate_dependent.  If an observable is ALSO reachable via a global
    // path, the global path will add it normally.
    //
    // That list is the TAGGED observables only (issue #38).  A bare
    // observable inside a local function keeps its global value, so it
    // falls through to the worklist here and is marked rate_dependent —
    // which is what keeps it refreshed after every event rather than only
    // at sample points.  Misclassifying it as local would have skipped
    // that refresh too, so a fix to the local-scope override alone could
    // still have read a stale global value.
    std::unordered_set<std::string> rate_vars;
    std::vector<std::string> worklist(seeds.begin(), seeds.end());
    while (!worklist.empty()) {
      std::string const name = std::move(worklist.back());
      worklist.pop_back();
      if (!rate_vars.insert(name).second)
        continue; // already visited
      auto it = func_deps.find(name);
      if (it != func_deps.end()) {
        // Check if this is a local function — if so, skip its
        // locally-evaluated observable dependencies.
        std::unordered_set<std::string> local_obs_set;
        auto fi = model.function_index.find(name);
        if (fi != model.function_index.end()) {
          auto& gf = model.functions[fi->second];
          if (gf.is_local()) {
            for (auto& on : gf.local_observable_names)
              local_obs_set.insert(on);
          }
        }
        for (auto& dep : it->second) {
          if (!local_obs_set.empty() && local_obs_set.count(dep))
            continue; // locally evaluated — skip
          worklist.push_back(dep);
        }
      }
    }

    // Mark observables
    for (auto& obs : model.observables) {
      if (rate_vars.count(obs.name))
        obs.rate_dependent = true;
    }
  }

  // ---- 8b. Local rates that track a moving global (issue #38, #40) ----
  //
  // A local function built only from tagged observables and constants has
  // a per-instance value that can only change when that instance's own
  // neighbourhood changes — which is exactly what the engine's
  // affected-molecule delta path already covers.  Let a bare observable in
  // (or `time`, or a global function over either) and that stops holding:
  // every instance's rate moves at once, with no molecule marked affected.
  // Flag those rules so the engine rescans them after each event instead.
  //
  // Marking is deliberately narrow.  Every model in the three corpora
  // takes the constant-and-tagged-only shape, so none of them acquires a
  // per-event rescan from this.
  //
  // The walk also records WHAT the chain reads, not just that it reads
  // something (issue #40).  "Could this rule read a moving global?" is the
  // right question for choosing the rescan path and the wrong one for
  // running it every event: a bare observable is usually a volume proxy or
  // a total that never moves, and then every one of those O(N) rescans
  // recomputes rates that cannot have changed.  The engine compares these
  // resolved values against the previous rescan's and skips when they
  // agree — which is why the list is collected here, where the walk
  // already visits exactly those tokens.
  {
    // What a function's evaluation consults that a reaction event can
    // move.  `obs` are observable slots read at system scope; `time`
    // advances every event, and `opaque` marks a dependency this walk
    // cannot name — both mean "assume moved", i.e. rescan unconditionally.
    struct GlobalDeps {
      std::vector<int> obs;
      bool time = false;
      bool opaque = false;
      bool any() const { return !obs.empty() || time || opaque; }
      void merge(const GlobalDeps& o) {
        obs.insert(obs.end(), o.obs.begin(), o.obs.end());
        time = time || o.time;
        opaque = opaque || o.opaque;
      }
    };

    // Memoized and cycle-safe.  Re-entering a function still being walked
    // is a reference cycle, whose dependency set cannot be enumerated by
    // this walk (it would need its own fixed point) — so answer `opaque`
    // and let the engine rescan unconditionally.  The pre-#40 walk broke
    // the same cycle with a provisional `false`, which was safe when the
    // verdict only chose a path and is not safe now that a missing
    // observable would suppress a rescan.  Nothing in the corpora writes
    // one: a cyclic function chain has no value to evaluate at all.
    std::unordered_map<std::string, GlobalDeps> fn_deps;
    std::unordered_set<std::string> in_progress;
    auto global_deps = [&](const std::string& fname, auto&& self) -> GlobalDeps {
      auto fi = model.function_index.find(fname);
      if (fi == model.function_index.end())
        return {};
      if (auto memo = fn_deps.find(fname); memo != fn_deps.end())
        return memo->second;
      if (!in_progress.insert(fname).second) {
        GlobalDeps cyclic;
        cyclic.opaque = true;
        return cyclic;
      }
      const auto& gf = model.functions[fi->second];

      GlobalDeps deps;

      // A TFUN's counter is a time / observable / function value unless it
      // is a plain parameter, and the table turns it into a moving rate.
      // The table itself is fixed, so the counter's own dependencies are
      // the TFUN's: a table read at an unchanged counter returns an
      // unchanged value.
      if (gf.is_tfun && gf.tfun_counter_source != TfunCounterSource::Parameter) {
        switch (gf.tfun_counter_source) {
        case TfunCounterSource::Time:
          deps.time = true;
          break;
        case TfunCounterSource::Observable: {
          auto oi = model.observable_index.find(gf.tfun_counter_name);
          if (oi != model.observable_index.end())
            deps.obs.push_back(oi->second);
          else
            deps.opaque = true; // counter names an observable that isn't there
          break;
        }
        case TfunCounterSource::Function: {
          // Engine::get_tfun_counter_value evaluates this one at global
          // scope; a local callee has no tag there, so don't pretend to
          // know what it reads.
          auto ci = model.function_index.find(gf.tfun_counter_name);
          if (ci == model.function_index.end() || model.functions[ci->second].is_local())
            deps.opaque = true;
          else
            deps.merge(self(gf.tfun_counter_name, self));
          break;
        }
        default:
          deps.opaque = true; // is_tfun with no counter source RM can read
          break;
        }
      }

      for (const auto& tok : expr::collect_variables(gf.expression_text)) {
        // An observable tagged inside THIS function is per-instance and
        // already covered by the affected-molecule path; any other
        // observable reference is the system-wide count.  A parameter or
        // a builtin name matches none of the three and is constant.
        bool const is_time = (tok == "time" || tok == "t");
        auto const oi = model.observable_index.find(tok);
        bool const is_global_obs =
            oi != model.observable_index.end() &&
            std::find(gf.local_observable_names.begin(), gf.local_observable_names.end(), tok) ==
                gf.local_observable_names.end();
        if (is_time)
          deps.time = true;
        if (is_global_obs)
          deps.obs.push_back(oi->second);
        if (model.function_index.count(tok) != 0U)
          deps.merge(self(tok, self));
      }

      std::sort(deps.obs.begin(), deps.obs.end());
      deps.obs.erase(std::unique(deps.obs.begin(), deps.obs.end()), deps.obs.end());
      in_progress.erase(fname);
      fn_deps[fname] = deps;
      return deps;
    };

    // The DOR1 normalization clears the name on whichever side it turned
    // into the constant 1, so an empty name is exactly "no factor here".
    for (auto& rule : model.rules) {
      auto& rl = rule.rate_law;
      if (!rl.is_local && rl.type != RateLawType::FunctionProduct)
        continue;
      GlobalDeps deps;
      if (!rl.function_name.empty())
        deps.merge(global_deps(rl.function_name, global_deps));
      if (!rl.function_name_b.empty())
        deps.merge(global_deps(rl.function_name_b, global_deps));
      std::sort(deps.obs.begin(), deps.obs.end());
      deps.obs.erase(std::unique(deps.obs.begin(), deps.obs.end()), deps.obs.end());

      rl.local_rate_tracks_global = deps.any();
      rl.global_dep_observables = std::move(deps.obs);
      rl.global_dep_time = deps.time;
      rl.global_dep_opaque = deps.opaque;
    }
  }

  // ---- 8c. Rate laws that read `reactant_N()` (issue #59) ----
  //
  // The placeholder has no value of its own; it stands for the match count
  // of the rule's Nth reactant pattern, so it can only be resolved once we
  // know which rule is asking.  Record per rule the largest N its rate-law
  // function chain reads, which is what the engine needs to bind the slots
  // before evaluating the rate, and what the refusal below needs to check
  // that the rule actually has that many reactant patterns.
  //
  // The walk follows function references, so a rate law that reaches the
  // placeholder through a helper function is covered as well as the direct
  // `reactant_1()*reactant_2()*f()` that BNG2 writes for the idiom.
  {
    std::unordered_map<std::string, int> fn_max;
    std::unordered_set<std::string> in_progress;
    auto max_index = [&](const std::string& fname, auto&& self) -> int {
      auto fi = model.function_index.find(fname);
      if (fi == model.function_index.end())
        return 0;
      if (auto memo = fn_max.find(fname); memo != fn_max.end())
        return memo->second;
      if (!in_progress.insert(fname).second)
        return 0; // reference cycle: nothing more to learn down this arm
      const auto& gf = model.functions[fi->second];
      int best = gf.reactant_count_index;
      for (const auto& tok : expr::collect_variables(gf.expression_text))
        best = std::max(best, self(tok, self));
      in_progress.erase(fname);
      fn_max[fname] = best;
      return best;
    };

    for (auto& rule : model.rules) {
      auto& rl = rule.rate_law;
      int best = 0;
      if (!rl.function_name.empty())
        best = std::max(best, max_index(rl.function_name, max_index));
      if (!rl.function_name_b.empty())
        best = std::max(best, max_index(rl.function_name_b, max_index));
      rl.max_reactant_count_index = best;
    }
  }

  // Scan for unsupported features if requested
  if (unsupported_out) {
    *unsupported_out = scan_unsupported(*model_node);
    // Arrhenius energy rules whose shape RM does not expand (recorded by
    // try_expand_arrhenius above): the rule was silently NOT expanded, so
    // refuse the model rather than run it missing that reaction.
    for (const auto& rid : unsupported_arrhenius_ids)
      unsupported_out->push_back(
          {Severity::Error, "RateLaw@type=Arrhenius",
           "Arrhenius energy rule '" + rid +
               "' is not a supported 2-reactant binding/unbinding shape — RM "
               "(like NFsim) only expands energy binding rules. Coupled "
               "operations (state change / molecule add/delete), intramolecular "
               "ring-closure binding, same-type homodimer binding, >2-reactant "
               "rules, and rules with exclude/include constraints are deferred; "
               "the rule would be silently dropped. Pass --ignore-unsupported to "
               "run anyway without that rule."});
    for (const auto& what : mixed_scope_observables)
      unsupported_out->push_back(
          {Severity::Warn, "Function/Expression",
           "Observable " + what +
               " is referenced both applied to the function's tag and bare, i.e. "
               "at local and at global scope in one expression. RM evaluates it "
               "at local scope throughout; the bare reference reads the local "
               "value rather than the system-wide count."});
    // ERROR-level: a negative Michaelis constant (issue #46).  The law's
    // sFree is the positive root of a quadratic whose discriminant is
    // (S-Km-E)^2 + 4*Km*S; with Km < 0 that can go negative, and where it
    // does not the expression still yields a finite but meaningless rate, so
    // the rule would run on a number that means nothing.  Refuse rather than
    // simulate it.  Km is resolved through the parameter cascade here, so
    // this catches a constant and a derived parameter alike; a value that
    // arrives later through set_param / parameter_scan cannot be caught at
    // load, and is clamped to zero with a warning by set_rule_propensity.
    // `!(Km > 0) && Km != 0` refuses a negative Km and a NaN one (a parameter
    // expression can evaluate to NaN) while leaving Km == 0 alone — that one
    // is a removable singularity the engine reads as kcat*min(S,E).
    for (const auto& rule : model.rules) {
      double const km = rule.rate_law.mm_Km;
      if (rule.rate_law.type != RateLawType::MM || km == 0 || km > 0)
        continue;
      unsupported_out->push_back(
          {Severity::Error, "RateLaw@type=MM",
           "Rule '" + rule.id + "' (" + rule.name +
               ") has a Michaelis constant Km=" + std::to_string(km) +
               " — MM(kcat,Km) requires Km > 0, and a negative Km puts the rate law "
               "outside its domain (the rate is NaN wherever the discriminant goes "
               "negative, and meaningless where it does not). Fix the parameter. Pass "
               "--ignore-unsupported to run anyway; the rule's propensity is clamped "
               "to zero, so it will not fire."});
    }

    // The `TotalRate` keyword.  BNG2.pl does not implement it for network
    // simulations — RxnRule.pm carries the TODO saying it is "currently
    // implemented only for XML network-free output" — and generate_network
    // duly writes the rate law into the .net as an ordinary rate constant,
    // so the ODE integrates plain mass action and the observable crashes to
    // zero where NFsim holds it flat.  There is no BNG2 result to check a
    // TotalRate model against, which leaves NFsim as the only implementation.
    //
    // On most rules RM and NFsim agree, and RM reads the keyword the way
    // BNG2 documents it in RateLaw.pm ("If true, this ratelaw specifies the
    // Total reaction rate"): the propensity IS the rate law's value.  Those
    // rules warn, because nothing can verify them against the reference RM is
    // written against, but they run and they agree.
    //
    // They stop agreeing when a reactant pattern can match one molecule more
    // than once.  NFsim expands such a rule into one independent reaction
    // class per permutation (`_R1_sym1`, `_R1_sym2`, ...; see
    // NFinput.cpp::generateRxnPermutations).  For an ordinary rate law that
    // is correct — the matches partition across the classes and sum back —
    // but under TotalRate every class returns the WHOLE total rate, so the
    // rule runs at
    //
    //     rate x #{permutations whose reactant lists are all non-empty}
    //
    // Measured against `C(s) + D(t)` at kf: one free site 1.00x, two 2.02x,
    // three 2.98x, and 1.00x again when every C has the same slot pre-bound
    // (only one permutation populated).  That factor counts NFsim's internal
    // reaction classes rather than anything in the model — it is capped by
    // the permutation count no matter how many molecules exist, and it steps
    // down discretely as classes empty.  BNG2's network expansion, which
    // writes the statistical factor per SPECIES and live (`3*_rateLaw1`,
    // `2*_rateLaw1`, `_rateLaw1`), implies a third number again.  Three
    // readings, no oracle: RM refuses those rather than pick one silently.
    //
    // The test is structural and deliberately conservative: the rule is
    // refused when any reactant pattern touches a component whose molecule
    // type declares two or more of that name.  That is what lets a pattern
    // component land on more than one of the molecule's slots, which is
    // exactly the condition NFsim permutes over.  A refusal only has to
    // cover the divergence, so covering slightly more is safe; what it must
    // not do is fire on the ordinary shapes, and it does not — every
    // TotalRate rule in the NFsim validation models (v21-v26), in oscSystem,
    // and in the RuleHub examples names its components distinctly.
    for (const auto& rule : model.rules) {
      if (!rule.rate_law.is_total_rate)
        continue;

      std::string ambiguous;
      for (const auto& pm : rule.reactant_pattern.molecules) {
        if (pm.type_index < 0 || pm.type_index >= static_cast<int>(model.molecule_types.size()))
          continue;
        const auto& mt = model.molecule_types[pm.type_index];
        for (const auto& pc : pm.components) {
          int declared = 0;
          for (const auto& mtc : mt.components)
            if (mtc.name == pc.name)
              ++declared;
          if (declared >= 2) {
            ambiguous = pm.type_name + "(" + pc.name + ")";
            break;
          }
        }
        if (!ambiguous.empty())
          break;
      }

      if (!ambiguous.empty()) {
        unsupported_out->push_back(
            {Severity::Error, "RateLaw@totalrate",
             "Rule '" + rule.id + "' (" + rule.name +
                 ") uses the TotalRate keyword, and its reactant pattern " + ambiguous +
                 " can match one molecule in more than one way because the molecule type "
                 "declares several components of that name. RM and NFsim disagree on what "
                 "the rule's rate then is: NFsim expands the rule into one reaction per "
                 "symmetry permutation and gives each the whole total rate, so the rule "
                 "runs faster in proportion to the number of populated permutations, while "
                 "RM takes the rate law as the whole propensity. BioNetGen cannot settle it "
                 "-- it does not implement TotalRate for network simulations at all. Name "
                 "the components distinctly, or drop TotalRate and fold the reactant counts "
                 "into the rate law. Pass --ignore-unsupported to run anyway, on RM's "
                 "reading."});
        continue;
      }

      unsupported_out->push_back(
          {Severity::Warn, "RateLaw@totalrate",
           "Rule '" + rule.id + "' (" + rule.name +
               ") uses the TotalRate keyword. BioNetGen does not implement TotalRate for "
               "network simulations, only for the network-free XML it writes, so a model "
               "using it cannot be checked against BioNetGen's own result. RM reads the "
               "rate law as the whole propensity, which is what BioNetGen documents the "
               "keyword to mean and what NFsim also computes for this rule."});
    }

    // `reactant_N()` in a rate law (issue #59): two shapes RM cannot resolve,
    // and one it resolves in a way that is easy to write by accident.
    for (const auto& rule : model.rules) {
      int const n = rule.rate_law.max_reactant_count_index;
      if (n == 0)
        continue;

      // ERROR-level.  Both of these would leave the rule reading a
      // placeholder that never gets a value, i.e. a rate of zero and a rule
      // that never fires, which is the silent no-op this construct was
      // reported for in the first place.
      std::string reason;
      if (n > rule.molecularity) {
        reason = "reads reactant_" + std::to_string(n) + "(), but the rule has " +
                 std::to_string(rule.molecularity) +
                 " reactant pattern(s), so there is no such reactant to count";
      } else if (rule.rate_law.is_local || rule.rate_law.is_local_b) {
        reason = "reads reactant_" + std::to_string(n) +
                 "() from a rate law that is also a local (per-instance) function. "
                 "RM resolves reactant counts for whole-rule rate functions only";
      }
      if (!reason.empty()) {
        unsupported_out->push_back({Severity::Error, "RateLaw@type=Function",
                                    "Rule '" + rule.id + "' (" + rule.name + ") " + reason +
                                        ". Pass --ignore-unsupported to run anyway; the "
                                        "placeholder reads zero, so the rule will not fire."});
        continue;
      }

      // WARN-level: the counts land on the propensity twice.  A rate law
      // that is not a total rate states the rate PER set of reactants, so
      // the propensity is that value times the reactant match counts — and
      // `reactant_N()` supplies those same counts a second time, inside the
      // rate.  A bimolecular rule written `reactant_1()*reactant_2()*k` is
      // therefore priced at `(N1*N2)^2 * k`, not `N1*N2*k`.
      //
      // This is not a divergence: NFsim multiplies by the counts after
      // evaluating the rate function too, and RM matches it, which is why
      // this warns rather than refusing.  What it buys is that a reading
      // most people would not predict from the BNGL source is named at load
      // instead of found by wondering why a rate constant is off by orders
      // of magnitude.  `TotalRate` is the shape where the construct means
      // what it looks like, and it is silent there.
      if (!rule.rate_law.is_total_rate)
        unsupported_out->push_back(
            {Severity::Warn, "RateLaw@type=Function",
             "Rule '" + rule.id + "' (" + rule.name +
                 ") uses reactant_N() in its rate law and is not marked TotalRate. The "
                 "value of a rate function is multiplied by the rule's reactant match "
                 "counts to get the propensity, so the counts reactant_N() supplies are "
                 "applied a second time: this rule's propensity grows with the square of "
                 "each reactant count rather than in proportion to it. NFsim reads the "
                 "same model the same way, so RM is not diverging from it here. If the "
                 "rate law was meant to give the whole propensity on its own, mark the "
                 "rule TotalRate."});
    }

    // WARN-level: MM constructs where RM cannot reproduce BNG2 (issue #45).
    // BNG2.pl is the reference RM is written against, so both entries below
    // are divergences from it and both say so.  They warn rather than refuse
    // because the constructs are idiomatic BNGL — refusing would put a large
    // share of real MM models out of reach — and because RM's own reading is
    // well defined.  What the warning buys is that the divergence is named at
    // load rather than discovered by diffing trajectories.
    for (const auto& rule : model.rules) {
      if (rule.rate_law.type != RateLawType::MM)
        continue;

      // (a) A reactant pattern that can match more than one species.  BNG2's
      // network expansion emits one MM reaction per matching (substrate,
      // enzyme) species PAIR, each evaluating the law on that pair's own
      // counts, while RM applies one law to the summed match counts.
      // Measured: a substrate pattern matching two species runs 2.00x faster
      // under BNG2 in saturation (the factor is the number of matching
      // substrate species), and an ENZYME pattern matching two species runs
      // 1.81x faster with the enzyme in excess, since the law is not linear
      // in the enzyme count either.  Both axes vanish where the law is
      // linear.
      //
      // Matching BNG2 would need a live map from species canonical form to
      // match count on each slot, maintained per event so the law could be
      // evaluated once per species pair.  That is species-level bookkeeping
      // of the kind a network-free engine exists to avoid, and it costs most
      // on the very models that carry this construct: CaMKII_holo in the
      // reference corpus leaves six of seven components open on its
      // substrate pattern and has large complexes to canonicalize.  RM keeps
      // the pooled reading and names the divergence.
      //
      // BNG2 warns about the substrate axis at rule-read time via
      // checkSpeciesGraph(..., IsSpecies => 1) and the warning does not
      // survive into the XML, so RM recomputes the predicate: a pattern
      // matches at most one species iff every molecule specifies every
      // component its type declares, each with a definite state and a
      // definite bond status.  BNG2 runs that check on the substrate only,
      // so the enzyme axis is silent in BNG2 as well; RM checks both slots.
      {
        auto multi_species_reason = [&](int start, int end) -> std::string {
          for (int mi = start; mi < end; ++mi) {
            const auto& pm = rule.reactant_pattern.molecules[mi];
            if (pm.type_index < 0 || pm.type_index >= static_cast<int>(model.molecule_types.size()))
              continue;
            const auto& mt = model.molecule_types[pm.type_index];
            if (pm.components.size() < mt.components.size())
              return "molecule " + pm.type_name + " leaves " +
                     std::to_string(mt.components.size() - pm.components.size()) +
                     " of its components unspecified";
            for (const auto& pc : pm.components) {
              bool const stateful = pc.comp_type_index >= 0 &&
                                    pc.comp_type_index < static_cast<int>(mt.components.size()) &&
                                    mt.components[pc.comp_type_index].allowed_states.size() > 1;
              if (stateful && pc.required_state_index < 0)
                return "component " + pm.type_name + "." + pc.name + " has no definite state";
              if (pc.bond_constraint == BondConstraint::Wildcard ||
                  pc.bond_constraint == BondConstraint::Bound)
                return "component " + pm.type_name + "." + pc.name + " has no definite bond status";
            }
          }
          return {};
        };

        int const n_mols = static_cast<int>(rule.reactant_pattern.molecules.size());
        int const n_rp = static_cast<int>(rule.reactant_pattern_starts.size());
        for (int slot = 0; slot < n_rp && slot < 2; ++slot) {
          int const start = rule.reactant_pattern_starts[slot];
          int const end = (slot + 1 < n_rp) ? rule.reactant_pattern_starts[slot + 1] : n_mols;
          std::string const why = multi_species_reason(start, end);
          if (why.empty())
            continue;
          const char* const which = (slot == 0) ? "substrate" : "enzyme";
          unsupported_out->push_back(
              {Severity::Warn, "RateLaw@type=MM",
               "Rule '" + rule.id + "' (" + rule.name + ") has an MM(kcat,Km) rate law whose " +
                   which + " pattern can match more than one species (" + why +
                   "). BNG2 expands this into one MM reaction per matching species pair, each "
                   "evaluating the law on that pair's own counts, so its ODE/SSA runs faster "
                   "than RM (measured 2.00x for a two-species substrate in saturation, 1.81x "
                   "for a two-species enzyme with the enzyme in excess). RM applies one "
                   "Michaelis-Menten law to the summed match counts, since matching BNG2 would "
                   "require tracking the matching species individually. Enumerate them in "
                   "separate rules to get BNG2's reading, or write the enzyme mechanism "
                   "explicitly (S + E <-> SE -> P + E), which both engines agree on."});
        }
      }

      // (b) A symmetry_factor that cannot be attributed.  It belongs to the
      // reactant pattern the rule transforms (issue #37); when the rule
      // transforms BOTH, the scalar is a product of the two patterns' factors
      // and the XML gives one number with no way to split it.  RM applies the
      // whole factor to the substrate, which is right for the canonical shape
      // and exact wherever the law is linear (the rate goes as S*E there), but
      // up to 2x off in saturation if the symmetry was the enzyme slot's.
      if (rule.symmetry_factor != 1.0) {
        ReactantTransforms const rt = reactant_pattern_transforms(rule);
        bool const both_transformed = rt.resolvable && rt.transformed.size() > 1 &&
                                      rt.transformed[0] != 0 && rt.transformed[1] != 0;
        if (both_transformed)
          unsupported_out->push_back(
              {Severity::Warn, "ReactionRule@symmetry_factor",
               "Rule '" + rule.id + "' (" + rule.name +
                   ") has symmetry_factor=" + std::to_string(rule.symmetry_factor) +
                   " on an MM(kcat,Km) rate law and transforms both of its reactant "
                   "patterns, so the factor cannot be attributed to one of them from "
                   "the XML, the scalar being a product of both patterns\' factors. "
                   "RM applies it to the substrate count, which reproduces BNG2 for "
                   "the ordinary shape where the enzyme is a catalyst and anywhere "
                   "the law is linear. If the symmetry is the enzyme pattern\'s, RM "
                   "runs up to 2x fast against BNG2 in saturation."});
      }
    }
  }

  return model;
}

// ---------------------------------------------------------------------------
// Unsupported-feature scanner
// ---------------------------------------------------------------------------

bool has_child(const XmlNode& parent, const std::string& name) {
  return find_child(parent, name) != nullptr;
}

// Are the molecules of one <ReactantPattern> tied together by that pattern's
// own bonds?  A `.`-joined reactant whose molecules share no bond —
// `A(x).B(y)`, meaning "these molecules, anywhere in the same complex" — is a
// shape the n-ary sampler cannot place, since it reaches a pattern's non-seed
// molecules by following bonds out of the seed.  Mirrors nary_slot_connected()
// in engine.cpp.  A single-molecule pattern is trivially connected.
bool reactant_pattern_connected(const XmlNode& rp) {
  std::vector<std::string> mol_ids;
  if (auto* ml = find_child(rp, "ListOfMolecules")) {
    for (auto& mn : ml->children)
      if (mn.name == "Molecule")
        mol_ids.push_back(opt_attr(mn, "id"));
  }
  int const n = static_cast<int>(mol_ids.size());
  if (n <= 1)
    return true;

  // A component id is "<molecule id>_C<k>", so a bond endpoint belongs to the
  // longest molecule id it starts with (on an underscore boundary).
  auto mol_of_site = [&](const std::string& site) {
    int best = -1;
    for (int mi = 0; mi < n; ++mi) {
      const std::string& mid = mol_ids[mi];
      if (mid.empty() || site.size() <= mid.size())
        continue;
      if (site.compare(0, mid.size(), mid) != 0 || site[mid.size()] != '_')
        continue;
      if (best < 0 || mid.size() > mol_ids[best].size())
        best = mi;
    }
    return best;
  };

  std::vector<std::vector<int>> adj(n);
  if (auto* bl = find_child(rp, "ListOfBonds")) {
    for (auto& bn : bl->children) {
      if (bn.name != "Bond")
        continue;
      int const a = mol_of_site(opt_attr(bn, "site1"));
      int const b = mol_of_site(opt_attr(bn, "site2"));
      if (a >= 0 && b >= 0 && a != b) {
        adj[a].push_back(b);
        adj[b].push_back(a);
      }
    }
  }

  std::vector<char> seen(n, 0);
  std::vector<int> stack{0};
  seen[0] = 1;
  int reached = 1;
  while (!stack.empty()) {
    int const cur = stack.back();
    stack.pop_back();
    for (int const nb : adj[cur]) {
      if (seen[nb] != 0)
        continue;
      seen[nb] = 1;
      ++reached;
      stack.push_back(nb);
    }
  }
  return reached == n;
}

// Check if any ReactionRule has a non-empty attribute.
bool any_rule_has_attr(const XmlNode& model_node, const std::string& attr_name) {
  auto* rr_list = find_child(model_node, "ListOfReactionRules");
  if (!rr_list)
    return false;
  // The continue-then-test loop reads more clearly than std::any_of with
  // a multi-condition lambda predicate.  NOLINT(readability-use-anyofallof)
  for (auto& rr : rr_list->children) { // NOLINT(readability-use-anyofallof)
    if (rr.name != "ReactionRule")
      continue;
    auto it = rr.attributes.find(attr_name);
    if (it != rr.attributes.end() && !it->second.empty() && it->second != "0")
      return true;
  }
  return false;
}

// Check if any element in the tree has MoveConnected="1" attribute.
bool any_has_move_connected(const XmlNode& node) {
  auto it = node.attributes.find("MoveConnected");
  if (it != node.attributes.end() && it->second == "1")
    return true;
  // Recursive descent — std::any_of with a recursive lambda is awkward
  // and hides the call site.  NOLINT(readability-use-anyofallof)
  for (auto& child : node.children) { // NOLINT(readability-use-anyofallof)
    if (any_has_move_connected(child))
      return true;
  }
  return false;
}

// Find the first rule using a rate law of the given type.  Returns the
// rule id if found, empty string otherwise.  Used by scan_unsupported to
// flag the specific offending rule when refusing a model.
std::string first_rule_with_ratelaw_type(const XmlNode& model_node, const std::string& type_name) {
  auto* rr_list = find_child(model_node, "ListOfReactionRules");
  if (!rr_list)
    return "";
  for (auto& rr : rr_list->children) {
    if (rr.name != "ReactionRule")
      continue;
    auto* rl = find_child(rr, "RateLaw");
    if (!rl)
      continue;
    auto it = rl->attributes.find("type");
    if (it != rl->attributes.end() && it->second == type_name)
      return opt_attr(rr, "id");
  }
  return "";
}

std::vector<UnsupportedFeature> scan_unsupported(const XmlNode& model_node) {
  std::vector<UnsupportedFeature> warnings;

  // (exclude/include reactants/products are now supported — no error needed)

  // ERROR-level: compartments declared but RM does not implement
  // volume-based rate scaling.  Running such a model would silently
  // produce simulation output with incorrect bimolecular rates.
  if (has_child(model_node, "ListOfCompartments")) {
    auto* comp_list = find_child(model_node, "ListOfCompartments");
    bool has_compartments = false;
    if (comp_list) {
      for (auto& c : comp_list->children) {
        if (c.name == "Compartment") {
          has_compartments = true;
          break;
        }
      }
    }
    if (has_compartments)
      warnings.push_back({Severity::Error, "ListOfCompartments",
                          "Compartments declared — RM does not implement "
                          "compartment volume scaling; bimolecular reaction "
                          "rates would be silently incorrect. Pass "
                          "--ignore-unsupported to run anyway with a "
                          "well-mixed (volume=1) interpretation."});
  }

  // NOTE: eBNGL Arrhenius energy rules are handled in load_model, not here.
  // Supported 2-reactant binding rules are expanded (Sekar load-time
  // expansion — see energy_expand.hpp / try_expand_arrhenius); any shape RM
  // does not expand is recorded there and surfaced as a Tier-0 error, because
  // deciding support needs the fully-parsed rule (operation set, molecule
  // types, reactant-pattern membership) rather than a raw XML scan.  A bare
  // <ListOfEnergyPatterns> with `Function`-type rate laws (e.g.
  // isingspin_localfcn, feature_coverage/ft_energy_patterns) is not a trigger.

  // ERROR-level: legacy/unimplemented rate-law types.  BNG2 still parses
  // these but RM's rule loader (cpp/rulemonkey/simulator.cpp:~1157)
  // recognises only Ele, Function, MM, and FunctionProduct.  Anything else
  // falls through to the default rate_law (type=Ele, rate_value=0.0), so the
  // rule never fires — silently producing wrong trajectories.
  //
  //   Sat:             NFsim itself rejects this type explicitly with
  //                    "use MM instead"; we follow that policy.
  //   Hill:            no NFsim handler at all; only ODE/SSA networks.
  //   (FunctionProduct is now implemented — see RateLawType::FunctionProduct.)
  for (const auto& [type_name, advice] : std::initializer_list<std::pair<const char*, const char*>>{
           {"Sat", "Sat() is deprecated; rewrite the rule to use MM(kcat,Km) — "
                   "NFsim itself rejects Sat with the same recommendation."},
           {"Hill", "Hill() rate laws are network-only (no NFsim handler); "
                    "use generate_network() + simulate({method=>\"ode\"}) instead "
                    "of network-free simulation."}}) {
    auto rule_id = first_rule_with_ratelaw_type(model_node, type_name);
    if (!rule_id.empty()) {
      std::string const msg = "Rate law type '" + std::string(type_name) + "' on rule '" + rule_id +
                              "' — RM does not implement it; the rule would silently "
                              "have zero propensity. " +
                              advice +
                              " Pass --ignore-unsupported to run anyway (rule will not fire).";
      warnings.push_back({Severity::Error, "RateLaw@type=" + std::string(type_name), msg});
    }
  }

  // ERROR-level: an n-ary rule (>= 3 ReactantPatterns) in a shape the
  // engine's n-ary path does not implement (issues #24, #26).
  //
  // Rules of three or more reactant patterns are simulated when every
  // pattern is one connected piece — a single molecule or a bonded complex —
  // and the rate law is elementary; see the NaryState comment in engine.cpp
  // for the propensity and sampler.  The shapes below fall outside that and
  // would otherwise hit the two-slot machinery, whose slot B swallows
  // patterns 2..n into one bond-free pattern that scores zero embeddings for
  // free reactants.  The rule would then hold zero propensity and never
  // fire, with mass still conserved, so the trajectory looks valid unless
  // compared against another engine.
  //
  // This mirrors nary_shape_supported() in engine.cpp; the two must agree,
  // or a rule rejected there and not refused here goes silently inert again.
  constexpr int kMaxNaryPatterns = 6; // == engine.cpp's kMaxNarySlots
  if (auto* rr_list = find_child(model_node, "ListOfReactionRules")) {
    for (auto& rr : rr_list->children) {
      if (rr.name != "ReactionRule")
        continue;
      auto* rp_list = find_child(rr, "ListOfReactantPatterns");
      if (!rp_list)
        continue;

      int rp_count = 0;
      bool all_connected = true;
      for (auto& rpn : rp_list->children) {
        if (rpn.name != "ReactantPattern")
          continue;
        ++rp_count;
        if (!reactant_pattern_connected(rpn))
          all_connected = false;
      }
      if (rp_count < 3)
        continue;

      std::string rate_type = "Ele";
      if (auto* rl = find_child(rr, "RateLaw")) {
        auto it = rl->attributes.find("type");
        if (it != rl->attributes.end())
          rate_type = it->second;
      }

      // Each reason completes "Rule 'r' has N reactant patterns, ...".
      std::string reason;
      if (rp_count > kMaxNaryPatterns) {
        reason = "past the engine's n-ary limit of " + std::to_string(kMaxNaryPatterns);
      } else if (!all_connected) {
        reason = "one of them a disconnected complex — the n-ary path places a "
                 "pattern's non-seed molecules by following its bonds, so the "
                 "molecules of a `.`-joined reactant must be bonded to each other";
      } else if (rate_type != "Ele") {
        // Reachable in practice for `Function` (a local or global rate
        // function on a multi-reactant rule).  `MM` cannot get this far —
        // BNG2 itself refuses to write XML for it: "Michaelis-Menton type
        // ratelaw require exactly 2 reactants".
        reason = "under a '" + rate_type +
                 "' rate law — the n-ary path implements elementary "
                 "(mass-action) rates only";
      } else {
        continue; // supported — the engine simulates this rule
      }

      auto rule_name = opt_attr(rr, "name");
      if (rule_name.empty())
        rule_name = opt_attr(rr, "id");
      std::string msg = "Rule '";
      msg += rule_name;
      msg += "' has ";
      msg += std::to_string(rp_count);
      msg += " reactant patterns, ";
      msg += reason;
      msg += ". The rule would silently have zero propensity and never fire, while "
             "the rest of the model simulates normally. Rewrite it as a sequence of "
             "at most bimolecular steps (e.g. A + A -> A2 followed by A2 + A -> P). "
             "Pass --ignore-unsupported to run anyway (this rule will not fire).";
      warnings.push_back({Severity::Error, "ListOfReactantPatterns", msg});
    }
  }

  // ERROR-level: BNGL `population` keyword (hybrid particle-population SSA,
  // Hogg 2013).  NFsim treats `population`-typed molecule types as bulk
  // counters rather than tracked individuals, with restricted semantics
  // (one molecule per species, populations cannot bind to particles).
  // RM has no equivalent — the keyword shows up as `population="1"` on
  // the MoleculeType XML element and would be silently ignored, causing
  // RM to instantiate population types as ordinary particles. For models
  // with high population counts this both blows up memory and produces
  // trajectories that don't match NFsim's bulk-counter semantics.
  if (auto* mt_list = find_child(model_node, "ListOfMoleculeTypes")) {
    for (auto& mt : mt_list->children) {
      if (mt.name != "MoleculeType")
        continue;
      auto it = mt.attributes.find("population");
      if (it != mt.attributes.end() && it->second == "1") {
        std::string const mt_name = opt_attr(mt, "id");
        warnings.push_back({Severity::Error, "MoleculeType@population",
                            "MoleculeType '" + mt_name +
                                "' declared with `population` keyword — RM does not "
                                "implement hybrid particle-population SSA; this molecule "
                                "type would be silently treated as ordinary particles, "
                                "producing trajectories that diverge from NFsim's "
                                "bulk-counter semantics. Pass --ignore-unsupported to "
                                "run anyway with the population type treated as "
                                "particles (slow on high counts; trajectory will differ)."});
      }
    }
  }

  // ERROR-level: Fixed species ($-prefixed seed species) outside the
  // currently-implemented scope: single-molecule pattern, no bonds,
  // at most one Fixed per MoleculeType.  Anything outside that scope
  // (complex-fixed with bonds, duplicate-type Fixed) would require
  // pattern-based re-instantiation that RM does not currently
  // implement.
  if (auto* sp_list = find_child(model_node, "ListOfSpecies")) {
    std::unordered_map<std::string, int> fixed_type_counts;
    for (auto& spn : sp_list->children) {
      if (spn.name != "Species")
        continue;
      auto it = spn.attributes.find("Fixed");
      if (it == spn.attributes.end() || it->second != "1")
        continue;
      auto sp_name = opt_attr(spn, "name");

      // Count molecules and bonds inside this fixed species.
      int n_mol = 0, n_bond = 0;
      std::string mol_type_name;
      if (auto* ml = find_child(spn, "ListOfMolecules")) {
        for (auto& mn : ml->children) {
          if (mn.name != "Molecule")
            continue;
          ++n_mol;
          if (n_mol == 1)
            mol_type_name = opt_attr(mn, "name");
        }
      }
      if (auto* bl = find_child(spn, "ListOfBonds")) {
        for (auto& bn : bl->children) {
          if (bn.name == "Bond")
            ++n_bond;
        }
      }

      if (n_mol != 1 || n_bond != 0) {
        std::string const msg = "Fixed species '" + sp_name +
                                "' is multi-molecule or bonded (mols=" + std::to_string(n_mol) +
                                ", bonds=" + std::to_string(n_bond) +
                                ") — RM currently "
                                "supports only single-molecule Fixed species with no bonds. Pass "
                                "--ignore-unsupported to run with Fixed enforcement DISABLED "
                                "(this species would behave as if the `$` were absent, which "
                                "silently diverges from BNG2 ODE semantics).";
        warnings.push_back({Severity::Error, "Species@Fixed", msg});
        continue;
      }
      if (++fixed_type_counts[mol_type_name] > 1) {
        std::string const msg = "Multiple Fixed species declared for "
                                "MoleculeType '" +
                                mol_type_name +
                                "' — RM currently allows at "
                                "most one Fixed species per MoleculeType to avoid "
                                "matching overlap. Pass --ignore-unsupported to run with "
                                "Fixed enforcement DISABLED for the duplicate declarations.";
        warnings.push_back({Severity::Error, "Species@Fixed", msg});
      }
    }
  }

  if (any_has_move_connected(model_node))
    warnings.push_back(
        {Severity::Warn, "MoveConnected", "MoveConnected keyword — requires compartments"});

  if (any_rule_has_attr(model_node, "priority"))
    warnings.push_back(
        {Severity::Warn, "priority", "Rule priority modifier — execution order ignored"});

  return warnings;
}

// ---------------------------------------------------------------------------
// parameter_scan / bifurcate helpers
// ---------------------------------------------------------------------------

// Validates the non-range parts of a ScanSpec common to parameter_scan and
// bifurcate.  `who` names the caller for the error text.  Range validation
// (par_min/par_max/n_points/log_scale) lives in build_scan_values.
void validate_scan_spec(const ScanSpec& spec, bool session_active,
                        const std::unordered_map<std::string, double>& base_parameters,
                        const char* who) {
  if (session_active)
    throw std::runtime_error(std::string("Cannot ") + who + " during active session");
  if (spec.parameter.empty())
    throw std::runtime_error(std::string(who) + ": spec.parameter is empty");
  if (!base_parameters.count(spec.parameter))
    throw std::runtime_error(std::string(who) + ": unknown parameter '" + spec.parameter +
                             "' (must be a parameter declared in the loaded XML)");
  if (spec.per_point.n_points < 1)
    throw std::runtime_error(std::string(who) +
                             ": per_point.n_points must be >= 1 (need at least one "
                             "sampled segment to record an endpoint)");
  if (spec.per_point.t_end < spec.per_point.t_start)
    throw std::runtime_error(std::string(who) + ": per_point.t_end (" +
                             std::to_string(spec.per_point.t_end) + ") is earlier than t_start (" +
                             std::to_string(spec.per_point.t_start) + ")");
}

// Resolves a ScanSpec's swept-parameter values.  An explicit `values` list
// takes precedence; otherwise a range is generated from par_min/par_max/
// n_points with optional geometric (log) spacing.  Mirrors BNG's
// par_scan_vals vs par_min/par_max/n_scan_pts precedence and validation.
std::vector<double> build_scan_values(const ScanSpec& spec) {
  if (!spec.values.empty())
    return spec.values;

  if (spec.n_points < 1)
    throw std::runtime_error("parameter_scan: n_points must be >= 1 when no explicit "
                             "value list is given");
  const bool degenerate = (spec.par_min == spec.par_max);
  if (!degenerate && spec.n_points < 2)
    throw std::runtime_error("parameter_scan: n_points must be > 1 when par_min != par_max");
  if (spec.log_scale && (spec.par_min <= 0.0 || spec.par_max <= 0.0))
    throw std::runtime_error("parameter_scan: log_scale requires par_min and par_max > 0");

  std::vector<double> values;
  values.reserve(static_cast<std::size_t>(spec.n_points));
  if (spec.n_points == 1 || degenerate) {
    // A single point (or a zero-width range) is just par_min repeated;
    // sidesteps the (n_points - 1) division below.
    for (int k = 0; k < spec.n_points; ++k)
      values.push_back(spec.par_min);
    return values;
  }

  const double lo = spec.log_scale ? std::log(spec.par_min) : spec.par_min;
  const double hi = spec.log_scale ? std::log(spec.par_max) : spec.par_max;
  const double delta = (hi - lo) / (spec.n_points - 1);
  for (int k = 0; k < spec.n_points; ++k) {
    const double x = lo + (k * delta);
    values.push_back(spec.log_scale ? std::exp(x) : x);
  }
  return values;
}

} // anonymous namespace

// ===========================================================================
// RuleMonkeySimulator pImpl
// ===========================================================================

struct RuleMonkeySimulator::Impl {
  Model model;
  Method method = Method::NfExact;
  std::string xml_path_str;
  std::unordered_map<std::string, double> param_overrides;
  // Direct seed-species amount overrides, keyed by index into
  // `model.initial_species` (issue #23).  Applied by apply_overrides
  // after the concentration-expression walk, so they take precedence
  // over whatever a parameter override derives for the same species.
  std::map<int, double> initial_amount_overrides;
  // True while apply_overrides has left overridden numbers baked into
  // the parsed model, so a later apply with no overrides still knows it
  // must run one restoring pass.
  bool overrides_applied_ = false;
  int molecule_limit = -1;

  std::unique_ptr<Engine> session;

  std::vector<std::string> obs_names;
  std::vector<std::string> param_names;
  std::vector<std::string> func_names; // global (non-local) function names
  std::vector<UnsupportedFeature> unsupported_features;

  // Keep a clean copy of parameters for override/restore
  std::unordered_map<std::string, double> base_parameters;

  // ExprTk evaluator for the parameter cascade + Ele/MM rate constants +
  // initial-species concentrations (issue #6).  Its variables are bound
  // by address into `model.parameters` (see build_param_evaluator), so
  // model.parameters keys must never be inserted/erased/whole-map-
  // reassigned after build_param_evaluator() runs — sync_parameters
  // mutates values in place for exactly this reason.  `param_eval_ids_`
  // memoizes compiled-expression ids keyed by the raw expression string.
  bngsim::ExprTkEvaluator param_eval_;
  std::unordered_map<std::string, int> param_eval_ids_;

  // Bind every model parameter into `param_eval_` by address.  Idempotent:
  // re-initialize replaces `model`, so the evaluator is rebuilt from
  // scratch each call (a fresh ExprTkEvaluator drops all prior bindings).
  void build_param_evaluator() {
    param_eval_ = bngsim::ExprTkEvaluator{};
    param_eval_ids_.clear();
    for (auto& name : model.parameter_names_ordered)
      param_eval_.define_variable(name, &model.parameters[name]);
  }

  // Symbolic-cascade baseline: what each parameter's `<Parameter expr=>`
  // resolves to with NO overrides in force, computed against the loaded
  // `value` numbers.  Memoized on first use — it depends only on the
  // parsed model.  A parameter whose expr fails to compile gets no entry
  // and is skipped by the cascade below.
  //
  // This is the gate that lets sync_parameters re-derive from `expr`
  // without perturbing anything an override does not actually touch: a
  // parameter whose expr-resolved value is unmoved from this baseline
  // keeps its loaded `value`, digit for digit.  Without it, merely
  // calling set_param on an unrelated parameter would silently re-round
  // every derived quantity in the model to expr precision (issue #23).
  std::unordered_map<std::string, double> symbolic_base_;
  bool symbolic_base_ready_ = false;

  // Populate `symbolic_base_`.  MUST be called with model.parameters
  // holding the un-overridden base values, which is exactly the state
  // sync_parameters is in right after its reset-to-base loop.
  void build_symbolic_baseline() {
    if (symbolic_base_ready_)
      return;
    symbolic_base_ready_ = true;
    for (const auto& [name, expr] : model.parameter_expr_attrs) {
      try {
        symbolic_base_[name] = resolve_cached(expr, param_eval_, param_eval_ids_);
      } catch (...) { // NOLINT(bugprone-empty-catch)
        // Unparseable expr — leave the parameter on its `value` source.
      }
    }
  }

  // Rebuild model.parameters from base_parameters + param_overrides,
  // cascading derived parameter expressions so an override on a base
  // parameter propagates to any parameter that references it
  // (e.g., `B = 2*A` recomputes when A is overridden).
  //
  // Cheap enough to call from set_param / clear_param_overrides so
  // get_parameter() returns a coherent view between runs without
  // requiring a full apply_overrides() rate-law / species walk.
  void sync_parameters() {
    // Reset to base values IN PLACE.  `param_eval_` binds its variables
    // by address into `model.parameters`, so the map's nodes must not be
    // relocated — a whole-map `model.parameters = base_parameters` could
    // do exactly that.  model.parameters and base_parameters carry the
    // identical key set (parameters are never added/removed after load),
    // so an in-place value overwrite is sufficient.
    for (auto& [name, val] : model.parameters) {
      auto bit = base_parameters.find(name);
      if (bit != base_parameters.end())
        val = bit->second;
    }
    // Snapshot the expr-resolved baseline while the map still holds base
    // values (see build_symbolic_baseline).  Only needed once there is
    // something to cascade.
    if (!param_overrides.empty())
      build_symbolic_baseline();

    for (auto& [name, val] : param_overrides) {
      auto it = model.parameters.find(name);
      if (it != model.parameters.end())
        it->second = val;
    }

    // Re-cascade in declaration order: a parameter not directly
    // overridden re-resolves its parsed expression against the
    // current (overridden) map.  Overridden parameters keep their
    // override regardless of expression.
    //
    // Two sources are in play per parameter.  `<Parameter expr=>` is the
    // symbolic derivation and is the only one that can propagate an
    // override; `<Parameter value=>` is BNG2's already-collapsed number,
    // which re-resolves to itself and therefore cannot.  Prefer `expr`,
    // but only where it moves the parameter off its symbolic baseline —
    // i.e. where the override genuinely reaches it.  Everything else
    // keeps the loaded `value` byte for byte, so a model simulated with
    // an override on parameter X is numerically identical to the
    // un-overridden model everywhere X does not reach (issue #23).
    //
    // Iterate to fixed point so a chain `C = 2*B; B = 2*A; A = ...`
    // declared in NON-dependency order still settles after a
    // set_param("A", x) — matches load_model's parse-time fixed-point
    // resolution.  BNG2 emits parameters in dependency order so a
    // single pass settles in practice, but hand-crafted XML or a
    // future emitter shouldn't silently produce a stale derived
    // value.  Cap at param_count + 4 to bound the work and bail on
    // dependency cycles (which can't resolve regardless).
    const int max_passes = static_cast<int>(model.parameter_names_ordered.size()) + 4;
    bool hit_cap = false;
    for (int pass = 0; pass < max_passes; ++pass) {
      bool changed = false;
      for (auto& name : model.parameter_names_ordered) {
        if (param_overrides.count(name))
          continue;
        try {
          double resolved = 0.0;
          auto sit = symbolic_base_.find(name);
          if (sit != symbolic_base_.end()) {
            const double from_expr =
                resolve_cached(model.parameter_expr_attrs.at(name), param_eval_, param_eval_ids_);
            // Unmoved from baseline => no override reaches this
            // parameter => keep the loaded `value`, not the expr
            // re-rounding of it.
            resolved = (from_expr == sit->second) ? base_parameters.at(name) : from_expr;
          } else {
            auto eit = model.parameter_value_attrs.find(name);
            if (eit == model.parameter_value_attrs.end())
              continue;
            resolved = resolve_cached(eit->second, param_eval_, param_eval_ids_);
          }
          if (resolved != model.parameters[name]) {
            model.parameters[name] = resolved;
            changed = true;
          }
        } catch (...) { // NOLINT(bugprone-empty-catch)
          // resolution failure leaves the prior value in place
        }
      }
      if (!changed) {
        break;
      }
      if (pass == max_passes - 1) {
        hit_cap = true;
      }
    }
    if (hit_cap) {
      std::fprintf(stderr,
                   "Warning: parameter override cascade did not converge after %d passes "
                   "(parameter count = %zu); a dependency cycle is likely. "
                   "Stale parameter values may be used.\n",
                   max_passes, model.parameter_names_ordered.size());
    }

    sync_initial_amounts();
  }

  // Re-resolve every seed-species amount against the current parameter
  // map, then re-apply the direct pins on top.  Called from
  // sync_parameters so `SpeciesInit::concentration` stays coherent with
  // `model.parameters` between runs, the same contract get_parameter()
  // already offers for parameters — which is what lets initial_species()
  // and get_initial_amount() answer without a run in between.
  void sync_initial_amounts() {
    for (auto& si : model.initial_species) {
      if (!si.concentration_expr.empty())
        si.concentration = resolve_cached(si.concentration_expr, param_eval_, param_eval_ids_);
    }
    // Direct seed-amount overrides win over the re-resolved expression:
    // set_initial_amount is the escape hatch for amounts that no
    // parameter drives (a literal `concentration="1000"`), so it must not
    // be undone by the walk above (issue #23).
    for (const auto& [idx, amount] : initial_amount_overrides)
      model.initial_species[idx].concentration = amount;
  }

  // Resolve a caller-supplied seed-species key to an index into
  // `model.initial_species`.  Matches the BNGL `<Species name=>` pattern
  // first — that is what initial_species() reports and what a modeller
  // recognises — then falls back to the XML `<Species id=>` ("S1").
  int resolve_initial_species(const std::string& key, const char* caller) const {
    int by_name = -1;
    for (int i = 0; i < static_cast<int>(model.initial_species.size()); ++i) {
      if (model.initial_species[i].name != key)
        continue;
      if (by_name >= 0)
        throw std::runtime_error(std::string(caller) + ": seed species name '" + key +
                                 "' is declared more than once; use the XML <Species id=> instead");
      by_name = i;
    }
    if (by_name >= 0)
      return by_name;
    for (int i = 0; i < static_cast<int>(model.initial_species.size()); ++i) {
      if (model.initial_species[i].id == key)
        return i;
    }
    throw std::runtime_error(std::string(caller) + ": '" + key +
                             "' is not a seed species of the loaded XML (expected a <Species "
                             "name=> BNGL pattern or a <Species id=>)");
  }

  // Full override application: parameter cascade + re-resolve every
  // parsed-at-load-time numeric field that the engine reads directly
  // (Ele rate constants, MM kcat/Km, initial species concentrations,
  // Fixed-species target counts).  Called by run / initialize /
  // load_state immediately before the Engine snapshot is taken.
  void apply_overrides() {
    sync_parameters();

    const bool have_overrides = !param_overrides.empty() || !initial_amount_overrides.empty();

    // No overrides → no re-resolution to do.  `load_model`'s parse-time
    // cascade already resolved every Ele/MM rate value and initial-species
    // concentration against `base_parameters`, and `base_parameters ==
    // model.parameters` when `param_overrides.empty()` (sync_parameters is
    // called from set_param / clear_param_overrides to maintain that
    // invariant), so the walk below would just rewrite the same values.
    //
    // The `overrides_applied_` latch is what makes clearing an override
    // actually take effect: the walk below MUTATES the parsed model in
    // place, so if a prior apply wrote overridden numbers into
    // `rate_value` / `concentration`, skipping the walk now would leave
    // those stale — the run would keep simulating the cleared override
    // even though get_parameter() reports the restored value.  One more
    // pass against the (already restored) parameter map puts them back.
    if (!have_overrides && !overrides_applied_)
      return;
    overrides_applied_ = have_overrides;

    // Ele/MM rate values are baked from `model.parameters` at parse time.
    // Re-resolve them here so set_param overrides actually reach Engine
    // (which reads `rate_value` / `mm_kcat` / `mm_Km` directly for the
    // non-dynamic paths). Function rate laws evaluate against eval_vars
    // built live from `model.parameters`, so they need no fix-up.
    for (auto& rule : model.rules) {
      auto& rl = rule.rate_law;
      if (rl.type == RateLawType::Ele && !rl.rate_expr.empty())
        rl.rate_value = resolve_cached(rl.rate_expr, param_eval_, param_eval_ids_);
      else if (rl.type == RateLawType::MM) {
        if (!rl.mm_kcat_expr.empty())
          rl.mm_kcat = resolve_cached(rl.mm_kcat_expr, param_eval_, param_eval_ids_);
        if (!rl.mm_Km_expr.empty())
          rl.mm_Km = resolve_cached(rl.mm_Km_expr, param_eval_, param_eval_ids_);
      }
    }

    // Initial species concentrations are likewise baked at parse time
    // (Engine reads SpeciesInit::concentration during init_species), but
    // sync_parameters above already refreshed them via
    // sync_initial_amounts.  FixedSpecies::target_count is derived from
    // the same value during parse, so refresh it here from the
    // (possibly re-resolved) source.
    for (auto& fs : model.fixed_species) {
      if (fs.source_init_idx >= 0 &&
          fs.source_init_idx < static_cast<int>(model.initial_species.size())) {
        fs.target_count = static_cast<int>(model.initial_species[fs.source_init_idx].concentration);
      }
    }
  }

  // Endpoint observable / global-function values for one chain of scan
  // points.  Both matrices are row-major: `obs[point_idx][obs_idx]`.
  struct ScanChainEndpoints {
    std::vector<std::vector<double>> obs;
    std::vector<std::vector<double>> fun;
  };

  // Runs an ordered sequence of `(parameter = value)` scan points and
  // returns the endpoint (last sample row) of each point's trajectory.
  //
  // `reset_conc == true`: every point is an independent run() from the
  // model's seed species — the dose-response case.  `reset_conc == false`:
  // every point continues from the prior point's final molecular state,
  // threaded through a temporary state file, so the sweep is one
  // continuous-time trajectory sharing a single RNG stream.
  //
  // The swept override is restored to its pre-call state on the way out
  // (including the exception path), leaving the simulator exactly as the
  // caller configured it.  Assumes no session is active on entry.
  ScanChainEndpoints run_scan_chain(const std::string& parameter, const std::vector<double>& values,
                                    const TimeSpec& per_point, bool reset_conc,
                                    std::uint64_t seed) {
    // Snapshot the swept parameter's prior override so the chain is
    // side-effect free regardless of how it exits.
    const bool had_override = param_overrides.count(parameter) != 0;
    const double prior_value = had_override ? param_overrides.at(parameter) : 0.0;
    auto restore_override = [&]() {
      if (had_override)
        param_overrides[parameter] = prior_value;
      else
        param_overrides.erase(parameter);
      sync_parameters();
    };

    // One state file per chain invocation; reused (overwritten) across the
    // points of a carry-over chain, removed when the chain finishes.  A
    // process-global counter keeps concurrent chains from colliding.
    static std::atomic<unsigned long long> chain_counter{0};
    const std::filesystem::path tmp =
        std::filesystem::temp_directory_path() /
        ("rm_scan_" + std::to_string(chain_counter.fetch_add(1)) + ".rmstate");
    bool tmp_written = false;
    const double duration = per_point.t_end - per_point.t_start;

    ScanChainEndpoints out;
    out.obs.reserve(values.size());
    out.fun.reserve(values.size());

    try {
      for (std::size_t k = 0; k < values.size(); ++k) {
        param_overrides[parameter] = values[k];
        sync_parameters();
        apply_overrides();

        Result r;
        if (reset_conc) {
          // Fresh, independent run from seed species.  Every point uses
          // the same seed: as in BNG, the seed is run-level, not
          // per-point, so points share a random stream.
          Engine engine(model, seed, molecule_limit);
          r = engine.run(TimeSpec{per_point.t_start, per_point.t_end, per_point.n_points, {}});
        } else {
          // Carry-over chain: point 0 starts from seed species; every
          // later point resumes the prior point's pool + RNG state.
          if (k == 0) {
            session = std::make_unique<Engine>(model, seed, molecule_limit);
            session->initialize();
          } else {
            session = std::make_unique<Engine>(model, 0, molecule_limit);
            session->load_state(tmp.string());
          }
          const double t0 = session->current_time();
          r = session->run(TimeSpec{t0, t0 + duration, per_point.n_points, {}});
          if (k + 1 < values.size()) {
            session->save_state(tmp.string());
            tmp_written = true;
          }
          session.reset();
        }

        // Endpoint = the last sample row (the values at t_end).
        if (r.n_times() == 0)
          throw std::runtime_error("run_scan_chain: point produced no samples");
        std::vector<double> obs_row(r.n_observables());
        for (std::size_t o = 0; o < r.n_observables(); ++o)
          obs_row[o] = r.observable_data[o].back();
        std::vector<double> fun_row(r.n_functions());
        for (std::size_t f = 0; f < r.n_functions(); ++f)
          fun_row[f] = r.function_data[f].back();
        out.obs.push_back(std::move(obs_row));
        out.fun.push_back(std::move(fun_row));
      }
    } catch (...) {
      session.reset();
      if (tmp_written) {
        std::error_code ec;
        std::filesystem::remove(tmp, ec);
      }
      restore_override();
      throw;
    }

    if (tmp_written) {
      std::error_code ec;
      std::filesystem::remove(tmp, ec);
    }
    restore_override();
    return out;
  }
};

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

RuleMonkeySimulator::RuleMonkeySimulator(const std::string& xml_path, Method method) {
  if (xml_path.empty())
    throw std::runtime_error("XML path must not be empty");
  impl_ = std::make_unique<Impl>();
  impl_->method = method;
  impl_->xml_path_str = xml_path;
  impl_->model = load_model(xml_path, &impl_->unsupported_features);
  // Bind the parameter cascade evaluator onto the now-final model.
  // load_model resolved every parameter / Ele / MM / concentration once
  // already (its own load-time evaluator); param_eval_ takes over for
  // set_param re-resolution (sync_parameters / apply_overrides).
  impl_->build_param_evaluator();
  impl_->base_parameters = impl_->model.parameters;
  impl_->obs_names = impl_->model.observable_names_ordered;
  impl_->param_names = impl_->model.parameter_names_ordered;
  // Global (non-local) functions only, and not the `reactant_N()`
  // placeholders, which have a value only inside a rule (issue #59) — must
  // use the same filter and declaration order as
  // Engine::output_function_names so the simulator's function_names()
  // agrees with a Result's function_names.
  for (const auto& gf : impl_->model.functions) {
    if (!gf.is_local() && gf.reactant_count_index == 0)
      impl_->func_names.push_back(gf.name);
  }
}

RuleMonkeySimulator::~RuleMonkeySimulator() = default;
RuleMonkeySimulator::RuleMonkeySimulator(RuleMonkeySimulator&&) noexcept = default;
RuleMonkeySimulator& RuleMonkeySimulator::operator=(RuleMonkeySimulator&&) noexcept = default;

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

std::vector<std::string> RuleMonkeySimulator::observable_names() const { return impl_->obs_names; }

std::vector<std::string> RuleMonkeySimulator::function_names() const { return impl_->func_names; }

std::vector<std::string> RuleMonkeySimulator::parameter_names() const { return impl_->param_names; }

const std::string& RuleMonkeySimulator::xml_path() const { return impl_->xml_path_str; }

Method RuleMonkeySimulator::method() const { return impl_->method; }

const std::vector<UnsupportedFeature>& RuleMonkeySimulator::unsupported_features() const {
  return impl_->unsupported_features;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

void RuleMonkeySimulator::set_param(const std::string& name, double value) {
  if (impl_->session)
    throw std::runtime_error("Cannot set_param during active session");
  if (!impl_->base_parameters.count(name))
    throw std::runtime_error("Unknown parameter '" + name +
                             "' (set_param only accepts names declared in the loaded XML)");
  impl_->param_overrides[name] = value;
  impl_->sync_parameters();
}

void RuleMonkeySimulator::clear_param_overrides() {
  if (impl_->session)
    throw std::runtime_error("Cannot clear_param_overrides during active session");
  impl_->param_overrides.clear();
  impl_->sync_parameters();
}

std::vector<InitialSpeciesRow> RuleMonkeySimulator::initial_species() const {
  std::vector<InitialSpeciesRow> rows;
  rows.reserve(impl_->model.initial_species.size());
  for (int i = 0; i < static_cast<int>(impl_->model.initial_species.size()); ++i) {
    const auto& si = impl_->model.initial_species[i];
    InitialSpeciesRow row;
    row.id = si.id;
    row.name = si.name;
    row.concentration_expr = si.concentration_expr;
    row.amount = si.concentration;
    row.overridden = impl_->initial_amount_overrides.count(i) != 0;
    rows.push_back(std::move(row));
  }
  return rows;
}

double RuleMonkeySimulator::get_initial_amount(const std::string& key) const {
  int const idx = impl_->resolve_initial_species(key, "get_initial_amount");
  return impl_->model.initial_species[idx].concentration;
}

void RuleMonkeySimulator::set_initial_amount(const std::string& key, double amount) {
  if (impl_->session)
    throw std::runtime_error("Cannot set_initial_amount during active session");
  if (!std::isfinite(amount) || amount < 0.0)
    throw std::runtime_error("set_initial_amount: amount for '" + key +
                             "' must be a finite, non-negative molecule count");
  int const idx = impl_->resolve_initial_species(key, "set_initial_amount");
  impl_->initial_amount_overrides[idx] = amount;
  impl_->sync_parameters();
}

void RuleMonkeySimulator::clear_initial_amount_overrides() {
  if (impl_->session)
    throw std::runtime_error("Cannot clear_initial_amount_overrides during active session");
  impl_->initial_amount_overrides.clear();
  impl_->sync_parameters();
}

void RuleMonkeySimulator::set_molecule_limit(int limit) {
  if (impl_->session)
    throw std::runtime_error("Cannot set_molecule_limit during active session");
  impl_->molecule_limit = limit;
}

void RuleMonkeySimulator::set_block_same_complex_binding(bool value) {
  if (impl_->session)
    throw std::runtime_error("Cannot set_block_same_complex_binding during active session");
  impl_->model.block_same_complex_binding = value;
}

// ---------------------------------------------------------------------------
// One-shot simulation
// ---------------------------------------------------------------------------

Result RuleMonkeySimulator::run(const TimeSpec& ts, std::uint64_t seed,
                                const CancelCallback& should_continue) {
  impl_->apply_overrides();
  Engine engine(impl_->model, seed, impl_->molecule_limit);
  return engine.run(ts, should_continue);
}

// ---------------------------------------------------------------------------
// Parameter sweeps
// ---------------------------------------------------------------------------

ScanResult RuleMonkeySimulator::parameter_scan(const ScanSpec& spec, std::uint64_t seed) {
  validate_scan_spec(spec, impl_->session != nullptr, impl_->base_parameters, "parameter_scan");
  const std::vector<double> values = build_scan_values(spec);

  auto ep = impl_->run_scan_chain(spec.parameter, values, spec.per_point, spec.reset_conc, seed);

  ScanResult res;
  res.parameter = spec.parameter;
  res.param_values = values;
  res.observable_names = impl_->obs_names;
  res.function_names = impl_->func_names;
  res.observable_endpoints = std::move(ep.obs);
  res.function_endpoints = std::move(ep.fun);
  return res;
}

BifurcateResult RuleMonkeySimulator::bifurcate(const ScanSpec& spec, std::uint64_t seed) {
  validate_scan_spec(spec, impl_->session != nullptr, impl_->base_parameters, "bifurcate");
  const std::vector<double> ascending = build_scan_values(spec);
  const std::size_t n = ascending.size();

  // One continuous trajectory: the ascending forward sweep immediately
  // followed by the descending backward sweep.  The backward sweep's
  // first point repeats par_max, exactly as BNG's bifurcate re-runs the
  // endpoint without resetting concentrations.
  std::vector<double> sequence = ascending;
  sequence.reserve(2 * n);
  for (std::size_t i = n; i-- > 0;)
    sequence.push_back(ascending[i]);

  // Carry-over is intrinsic to a bifurcation sweep; spec.reset_conc is
  // ignored.
  auto ep =
      impl_->run_scan_chain(spec.parameter, sequence, spec.per_point, /*reset_conc=*/false, seed);

  BifurcateResult out;
  for (ScanResult* branch : {&out.forward, &out.backward}) {
    branch->parameter = spec.parameter;
    branch->param_values = ascending;
    branch->observable_names = impl_->obs_names;
    branch->function_names = impl_->func_names;
    branch->observable_endpoints.resize(n);
    branch->function_endpoints.resize(n);
  }
  for (std::size_t i = 0; i < n; ++i) {
    out.forward.observable_endpoints[i] = std::move(ep.obs[i]);
    out.forward.function_endpoints[i] = std::move(ep.fun[i]);
    // Backward run order is sequence[n .. 2n-1] = par_max .. par_min;
    // re-index to the same ascending axis as forward.
    out.backward.observable_endpoints[i] = std::move(ep.obs[(2 * n) - 1 - i]);
    out.backward.function_endpoints[i] = std::move(ep.fun[(2 * n) - 1 - i]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Session API
// ---------------------------------------------------------------------------

void RuleMonkeySimulator::initialize(std::uint64_t seed) {
  impl_->apply_overrides();
  impl_->session = std::make_unique<Engine>(impl_->model, seed, impl_->molecule_limit);
  impl_->session->initialize();
}

void RuleMonkeySimulator::step_to(double time, const CancelCallback& should_continue) {
  if (!impl_->session)
    throw std::runtime_error("No active session (call initialize first)");
  TimeSpec ts;
  ts.t_start = impl_->session->current_time();
  ts.t_end = time;
  ts.n_points = 0;
  // step_to advances without sampling output — just run the SSA
  impl_->session->run(ts, should_continue);
}

Result RuleMonkeySimulator::simulate(double t_start, double t_end, int n_points,
                                     const CancelCallback& should_continue) {
  if (!impl_->session)
    throw std::runtime_error("No active session (call initialize first)");

  // The session has its own current_time (advanced by initialize, prior
  // simulate() / step_to(), or load_state).  The contract is that the
  // segment starts there; a caller passing a t_start that disagrees has
  // a bug — going backwards is meaningless, and a forward gap silently
  // discards the burn-in window.  Throw rather than produce a degenerate
  // trajectory.
  const double session_t = impl_->session->current_time();
  const double tol = 1e-9 * std::max(1.0, std::fabs(session_t));
  if (std::fabs(t_start - session_t) > tol) {
    std::ostringstream msg;
    msg << "simulate(t_start=" << t_start << ", t_end=" << t_end
        << "): t_start disagrees with current session time " << session_t
        << "; pass t_start = sim.current_time() (or call destroy_session/initialize"
           " to restart at 0)";
    throw std::runtime_error(msg.str());
  }
  if (t_end < session_t)
    throw std::runtime_error("simulate: t_end (" + std::to_string(t_end) +
                             ") is earlier than current session time (" +
                             std::to_string(session_t) + ")");

  TimeSpec ts;
  ts.t_start = t_start;
  ts.t_end = t_end;
  ts.n_points = n_points;
  return impl_->session->run(ts, should_continue);
}

Result RuleMonkeySimulator::simulate(const TimeSpec& ts, const CancelCallback& should_continue) {
  if (!impl_->session)
    throw std::runtime_error("No active session (call initialize first)");

  // Like the (t_start, t_end, n_points) overload, the segment must start at
  // the current session time -- a disagreeing start is a caller bug (going
  // backwards is meaningless; a forward gap silently discards a burn-in
  // window).  With explicit ts.sample_times the segment start is the first
  // requested instant; otherwise it is ts.t_start.  The engine's run_ssa
  // honors ts.sample_times directly (records at exactly those sorted instants
  // in one pass), so no extra sampling logic is needed here.
  const double seg_start = ts.sample_times.empty() ? ts.t_start : ts.sample_times.front();
  const double session_t = impl_->session->current_time();
  const double tol = 1e-9 * std::max(1.0, std::fabs(session_t));
  if (std::fabs(seg_start - session_t) > tol) {
    std::ostringstream msg;
    msg << "simulate(TimeSpec): segment start " << seg_start
        << " disagrees with current session time " << session_t
        << "; pass the current session time as t_start / sample_times[0]"
           " (or destroy_session/initialize to restart at 0)";
    throw std::runtime_error(msg.str());
  }
  const double seg_end = ts.sample_times.empty() ? ts.t_end : ts.sample_times.back();
  if (seg_end < session_t)
    throw std::runtime_error("simulate(TimeSpec): segment end (" + std::to_string(seg_end) +
                             ") is earlier than current session time (" +
                             std::to_string(session_t) + ")");

  return impl_->session->run(ts, should_continue);
}

void RuleMonkeySimulator::save_state(const std::string& path) const {
  if (!impl_->session)
    throw std::runtime_error("No active session (call initialize first)");
  impl_->session->save_state(path);
}

void RuleMonkeySimulator::load_state(const std::string& path) {
  impl_->apply_overrides();
  // Create engine with seed=0 (will be overwritten by loaded RNG state)
  impl_->session = std::make_unique<Engine>(impl_->model, 0, impl_->molecule_limit);
  impl_->session->load_state(path);
}

bool RuleMonkeySimulator::has_session() const { return impl_->session != nullptr; }

double RuleMonkeySimulator::current_time() const {
  if (!impl_->session)
    throw std::runtime_error("No active session");
  return impl_->session->current_time();
}

void RuleMonkeySimulator::destroy_session() { impl_->session.reset(); }

// ---------------------------------------------------------------------------
// Session queries
// ---------------------------------------------------------------------------

std::vector<double> RuleMonkeySimulator::get_observable_values() {
  if (!impl_->session)
    throw std::runtime_error("No active session");
  return impl_->session->get_observable_values();
}

std::vector<double> RuleMonkeySimulator::get_function_values() {
  if (!impl_->session)
    throw std::runtime_error("No active session");
  return impl_->session->get_function_values();
}

double
RuleMonkeySimulator::evaluate_expression(const std::string& expr,
                                         const std::unordered_map<std::string, double>& extra) {
  if (!impl_->session)
    throw std::runtime_error("No active session");
  return impl_->session->evaluate_expression(expr, extra);
}

double RuleMonkeySimulator::get_parameter(const std::string& name) const {
  auto it = impl_->model.parameters.find(name);
  if (it == impl_->model.parameters.end())
    throw std::runtime_error("Unknown parameter '" + name + "'");
  return it->second;
}

int RuleMonkeySimulator::get_molecule_count(const std::string& type_name) const {
  if (!impl_->session)
    throw std::runtime_error("No active session");
  return impl_->session->get_molecule_count(type_name);
}

void RuleMonkeySimulator::add_molecules(const std::string& type_name, int count) {
  if (!impl_->session)
    throw std::runtime_error("No active session");
  impl_->session->add_molecules(type_name, count);
}

std::vector<SpeciesRow> RuleMonkeySimulator::enumerate_species() const {
  if (!impl_->session)
    throw std::runtime_error("No active session");
  return impl_->session->enumerate_species();
}

void RuleMonkeySimulator::write_species_file(const std::string& path) const {
  if (!impl_->session)
    throw std::runtime_error("No active session");
  impl_->session->write_species_file(path);
}

long RuleMonkeySimulator::species_count(const std::string& canonical_species) const {
  if (!impl_->session)
    throw std::runtime_error("No active session");
  return impl_->session->species_count(canonical_species);
}

long RuleMonkeySimulator::total_complex_count() const {
  if (!impl_->session)
    throw std::runtime_error("No active session");
  return impl_->session->total_complex_count();
}

// ---------------------------------------------------------------------------
// Pattern-keyed species methods (issue #9 §1)
// ---------------------------------------------------------------------------
// Each parses the runtime pattern string against the loaded molecule
// types — parse_species_pattern throws on a malformed / under-specified
// / wildcard pattern — then delegates the resolved Pattern to the
// engine.  The model the parser resolves against is the simulator's own
// copy; the engine holds an index-identical snapshot, so the resolved
// Pattern's type indices line up on both sides.

int RuleMonkeySimulator::get_species_count(const std::string& pattern) const {
  if (!impl_->session)
    throw std::runtime_error("No active session");
  const Pattern pat = parse_species_pattern(pattern, impl_->model);
  return static_cast<int>(impl_->session->get_species_count(pat));
}

void RuleMonkeySimulator::add_species(const std::string& pattern, int count) {
  if (!impl_->session)
    throw std::runtime_error("No active session");
  const Pattern pat = parse_species_pattern(pattern, impl_->model);
  impl_->session->add_species(pat, count);
}

void RuleMonkeySimulator::remove_species(const std::string& pattern, int count) {
  if (!impl_->session)
    throw std::runtime_error("No active session");
  const Pattern pat = parse_species_pattern(pattern, impl_->model);
  impl_->session->remove_species(pat, count);
}

void RuleMonkeySimulator::set_species_count(const std::string& pattern, int count) {
  if (!impl_->session)
    throw std::runtime_error("No active session");
  const Pattern pat = parse_species_pattern(pattern, impl_->model);
  impl_->session->set_species_count(pat, count);
}

} // namespace rulemonkey
