#include "rulemonkey/simulator.hpp"

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
      model.parameter_exprs[id] = val_str;
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
        const auto& val_str = model.parameter_exprs[name];
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
      const auto& val_str = model.parameter_exprs[name];
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
                        char* ea = nullptr;
                        char* eb = nullptr;
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

      // Parse references to identify locally-evaluated observables
      auto* ref_list = find_child(fn, "ListOfReferences");
      if (ref_list) {
        for (auto& rn : ref_list->children) {
          if (rn.name != "Reference")
            continue;
          auto rtype = opt_attr(rn, "type");
          if (rtype == "Observable") {
            gf.local_observable_names.push_back(need_attr(rn, "name"));
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

  // Scan for unsupported features if requested
  if (unsupported_out)
    *unsupported_out = scan_unsupported(*model_node);

  return model;
}

// ---------------------------------------------------------------------------
// Unsupported-feature scanner
// ---------------------------------------------------------------------------

bool has_child(const XmlNode& parent, const std::string& name) {
  return find_child(parent, name) != nullptr;
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

// True iff any rule uses an Arrhenius rate law — the eBNGL energy-based
// kinetic form that derives rate constants at runtime from free-energy
// sums over pattern matches.  RM does not implement that derivation.
bool any_rule_has_arrhenius_ratelaw(const XmlNode& model_node) {
  return !first_rule_with_ratelaw_type(model_node, "Arrhenius").empty();
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

  // ERROR-level: energy-based BNGL (eBNGL) — Arrhenius rate laws derive
  // their rate constants at runtime from a sum over energy-pattern
  // matches (ΔG computation per Sekar 2015).  RM does not implement
  // that derivation; running such a model would silently use whatever
  // fallback interpretation the rate-law parser applies, producing
  // incorrect trajectories.
  //
  // A bare <ListOfEnergyPatterns> is NOT a trigger on its own: many
  // BNGL models declare energy patterns purely for BNG2's ODE/SSA
  // generation and then use `Function`-type rate laws whose local
  // expressions inline the Boltzmann factors.  RM evaluates those
  // local functions faithfully, so the model simulates correctly
  // without runtime energy-pattern handling (e.g. isingspin_localfcn,
  // feature_coverage/ft_energy_patterns).  Only the Arrhenius rate
  // law needs runtime ΔG, so that is the precise trigger.
  if (any_rule_has_arrhenius_ratelaw(model_node)) {
    warnings.push_back({Severity::Error, "RateLaw@type=Arrhenius",
                        "Arrhenius rate law declared (eBNGL) — RM does not "
                        "implement runtime ΔG derivation from energy "
                        "patterns; rate constants would be silently "
                        "incorrect. Pass --ignore-unsupported to run "
                        "anyway with the Ea parameter treated as a bare "
                        "rate constant (no Boltzmann correction)."});
  }

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
        auto eit = model.parameter_exprs.find(name);
        if (eit == model.parameter_exprs.end())
          continue;
        try {
          double const resolved = resolve_cached(eit->second, param_eval_, param_eval_ids_);
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
  }

  // Full override application: parameter cascade + re-resolve every
  // parsed-at-load-time numeric field that the engine reads directly
  // (Ele rate constants, MM kcat/Km, initial species concentrations,
  // Fixed-species target counts).  Called by run / initialize /
  // load_state immediately before the Engine snapshot is taken.
  void apply_overrides() {
    sync_parameters();

    // No overrides → no re-resolution to do.  `load_model`'s parse-time
    // cascade already resolved every Ele/MM rate value and initial-species
    // concentration against `base_parameters`, and `base_parameters ==
    // model.parameters` when `param_overrides.empty()` (sync_parameters is
    // called from set_param / clear_param_overrides to maintain that
    // invariant), so the walk below would just rewrite the same values.
    if (param_overrides.empty())
      return;

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
    // (Engine reads SpeciesInit::concentration during init_species).
    // FixedSpecies::target_count is derived from the same value during
    // parse, so refresh it here from the (possibly re-resolved) source.
    for (auto& si : model.initial_species) {
      if (!si.concentration_expr.empty())
        si.concentration = resolve_cached(si.concentration_expr, param_eval_, param_eval_ids_);
    }
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
  // Global (non-local) functions only — must use the same filter and
  // declaration order as Engine::output_function_names so the simulator's
  // function_names() agrees with a Result's function_names.
  for (const auto& gf : impl_->model.functions) {
    if (!gf.is_local())
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
