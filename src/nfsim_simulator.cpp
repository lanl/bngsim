// bngsim/src/nfsim_simulator.cpp — NFsim stochastic simulator wrapper
//
// Wraps the vendored NFsim library for in-process rule-based simulation.
// Provides both stateless run() and a stateful session API for multi-action
// workflows (e.g., equilibrate → setConcentration → simulate).

#include "bngsim/nfsim_simulator.hpp"
#include "bngsim/expression.hpp"
#include "bngsim/platform_compat.hpp" // POSIX getpid() shim for Windows (GH #150)
#include "bngsim/result.hpp"
#include "bngsim/types.hpp"
#include "bngsim/wallclock.hpp"
#include "seed_count_rounding.hpp"

// NFsim headers (from vendored libnfsim)
#include "NFcore/NFcore.hh"
#include "NFcore/systemSnapshot.hh"
#include "NFfunction/NFfunction.hh"
#include "NFinput/NFinput.hh"
#include "NFutil/NFutil.hh"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace bngsim {

// ─── Thread-safe RAII cout/cerr suppressor ───────────────────────────────────
//
// NFsim prints copiously to cout and cerr during initialization and simulation.
// This RAII guard redirects both streams to a null buffer and restores them
// on destruction, even if an exception is thrown.
//
// Thread-safety: A static mutex serializes all NFsim operations that use
// StreamSuppressor. This prevents concurrent NFsim instances from corrupting
// each other's stream redirection. The mutex is acquired on construction and
// released on destruction, so the entire suppressed scope is serialized.
// ODE/SSA paths (which don't use StreamSuppressor) remain fully parallel.
//
// Note: NFsim's cout/cerr usage is too pervasive (100+ print sites in vendored
// code) to replace with a logging callback. Serialization is the pragmatic fix.
//
static std::mutex s_nfsim_stream_mutex;

class StreamSuppressor {
  public:
    StreamSuppressor()
        : lock_(s_nfsim_stream_mutex), old_cout_(std::cout.rdbuf()), old_cerr_(std::cerr.rdbuf()) {
        std::cout.rdbuf(null_stream_.rdbuf());
        std::cerr.rdbuf(null_stream_.rdbuf());
    }
    ~StreamSuppressor() {
        std::cout.rdbuf(old_cout_);
        std::cerr.rdbuf(old_cerr_);
        // lock_ released here by unique_lock destructor
    }

    std::string captured_output() const { return null_stream_.str(); }

    // Non-copyable
    StreamSuppressor(const StreamSuppressor &) = delete;
    StreamSuppressor &operator=(const StreamSuppressor &) = delete;

  private:
    std::unique_lock<std::mutex> lock_;
    std::streambuf *old_cout_;
    std::streambuf *old_cerr_;
    std::ostringstream null_stream_;
};

static std::string trim_ascii(std::string s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
        ++start;
    }
    if (start == s.size())
        return "";
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(start, end - start);
}

static std::string summarize_nfsim_log(const std::string &raw) {
    std::string trimmed = trim_ascii(raw);
    if (trimmed.empty())
        return "";

    constexpr size_t kMaxChars = 1200;
    if (trimmed.size() > kMaxChars) {
        trimmed = trimmed.substr(trimmed.size() - kMaxChars);
        trimmed = "[...]" + trimmed;
    }
    return trimmed;
}

// ─── Exact single-molecule BNGL species patterns ─────────────────────────────

enum class BondConstraint { Unbound, Bound, Any, Explicit };

struct ParsedComponentPattern {
    std::string name;
    std::optional<std::string> state;
    BondConstraint bond = BondConstraint::Unbound;
    int bond_id = -1;
};

struct ParsedMoleculePattern {
    std::string name;
    std::vector<ParsedComponentPattern> components;
};

struct ParsedSpeciesPattern {
    std::vector<ParsedMoleculePattern> molecules;
};

static std::vector<std::string> split_top_level(const std::string &s, char delimiter) {
    std::vector<std::string> out;
    std::string current;
    int depth = 0;
    for (char ch : s) {
        if (ch == '(') {
            ++depth;
        } else if (ch == ')') {
            --depth;
            if (depth < 0) {
                throw std::runtime_error("invalid BNGL pattern: unmatched ')'");
            }
        }

        if (ch == delimiter && depth == 0) {
            out.push_back(trim_ascii(current));
            current.clear();
        } else {
            current.push_back(ch);
        }
    }
    if (depth != 0) {
        throw std::runtime_error("invalid BNGL pattern: unmatched '('");
    }
    out.push_back(trim_ascii(current));
    return out;
}

static std::string normalize_molecule_name(std::string name) {
    name = trim_ascii(name);
    if (!name.empty() && name[0] == '@') {
        size_t scope = name.find("::");
        if (scope != std::string::npos) {
            name = name.substr(scope + 2);
        } else {
            scope = name.rfind(':');
            if (scope != std::string::npos) {
                name = name.substr(scope + 1);
            }
        }
    }
    return name;
}

static ParsedComponentPattern parse_component_pattern(const std::string &raw_component) {
    std::string text = trim_ascii(raw_component);
    if (text.empty()) {
        throw std::runtime_error("invalid BNGL pattern: empty component");
    }

    ParsedComponentPattern component;
    size_t pos = 0;
    while (pos < text.size() && text[pos] != '~' && text[pos] != '!') {
        if (std::isspace(static_cast<unsigned char>(text[pos]))) {
            throw std::runtime_error("invalid BNGL pattern: whitespace inside component");
        }
        ++pos;
    }
    component.name = text.substr(0, pos);
    if (component.name.empty()) {
        throw std::runtime_error("invalid BNGL pattern: component without name");
    }

    while (pos < text.size()) {
        char marker = text[pos++];
        size_t start = pos;
        while (pos < text.size() && text[pos] != '~' && text[pos] != '!') {
            ++pos;
        }
        std::string value = text.substr(start, pos - start);
        if (value.empty()) {
            throw std::runtime_error("invalid BNGL pattern: empty component modifier");
        }

        if (marker == '~') {
            if (component.state.has_value()) {
                throw std::runtime_error("invalid BNGL pattern: duplicate state modifier");
            }
            component.state = value;
        } else if (marker == '!') {
            if (value == "+") {
                component.bond = BondConstraint::Bound;
            } else if (value == "?") {
                component.bond = BondConstraint::Any;
            } else {
                int bond_id = 0;
                for (char ch : value) {
                    if (!std::isdigit(static_cast<unsigned char>(ch))) {
                        throw std::runtime_error("invalid BNGL pattern: unsupported bond token '!" +
                                                 value + "'");
                    }
                    bond_id = bond_id * 10 + (ch - '0');
                }
                component.bond = BondConstraint::Explicit;
                component.bond_id = bond_id;
            }
        }
    }

    return component;
}

static ParsedMoleculePattern parse_molecule_pattern(const std::string &raw_molecule) {
    std::string text = trim_ascii(raw_molecule);
    if (text.empty()) {
        throw std::runtime_error("invalid BNGL pattern: empty molecule");
    }

    ParsedMoleculePattern molecule;
    size_t open = text.find('(');
    if (open == std::string::npos) {
        molecule.name = normalize_molecule_name(text);
        return molecule;
    }

    if (text.back() != ')') {
        throw std::runtime_error("invalid BNGL pattern: molecule component list must end with ')'");
    }
    if (text.find('(', open + 1) != std::string::npos) {
        throw std::runtime_error("invalid BNGL pattern: nested molecule component list");
    }

    molecule.name = normalize_molecule_name(text.substr(0, open));
    std::string inner = text.substr(open + 1, text.size() - open - 2);
    inner = trim_ascii(inner);
    if (!inner.empty()) {
        for (const std::string &part : split_top_level(inner, ',')) {
            molecule.components.push_back(parse_component_pattern(part));
        }
    }
    return molecule;
}

static ParsedSpeciesPattern parse_species_pattern(const std::string &pattern) {
    ParsedSpeciesPattern parsed;
    std::string text = trim_ascii(pattern);
    if (text.empty()) {
        throw std::runtime_error("invalid BNGL pattern: empty pattern");
    }
    for (const std::string &part : split_top_level(text, '.')) {
        parsed.molecules.push_back(parse_molecule_pattern(part));
    }
    return parsed;
}

static int find_component_index(::NFcore::MoleculeType *mt, const std::string &name) {
    for (int i = 0; i < mt->getNumOfComponents(); ++i) {
        if (mt->getComponentName(i) == name) {
            return i;
        }
    }
    return -1;
}

static int find_state_value(::NFcore::MoleculeType *mt, int component_index,
                            const std::string &state_name) {
    auto possible_states = mt->getPossibleCompStates();
    if (component_index < 0 || component_index >= static_cast<int>(possible_states.size())) {
        return -1;
    }
    const auto &states = possible_states[component_index];
    for (int i = 0; i < static_cast<int>(states.size()); ++i) {
        if (states[i] == state_name) {
            return i;
        }
    }
    return -1;
}

struct ResolvedComponentConstraint {
    bool has_state = false;
    int state = ::NFcore::Molecule::NOSTATE;
    BondConstraint bond = BondConstraint::Unbound;
};

// One equivalency-class group within a resolved pattern.
//
// NFsim's XML loader renames duplicate <ComponentType id="r"> entries on a
// MoleculeType to internal names "r1", "r2", ... and remembers them as a
// symmetric class. A user-facing pattern like "L(r~u,r~u)" carries the
// bare name "r" twice; binding each parsed entry to a specific physical
// site index would over-constrain matching (NFsim treats both physical
// orderings of a class as the same species). Instead we resolve a class
// to: (a) its physical indices and (b) a *sorted multiset* of constraints
// that must be realized somewhere across those indices.
struct ResolvedClassGroup {
    std::vector<int> indices; // ascending
    std::vector<ResolvedComponentConstraint> sorted_constraints;
    bool stateful = false;
};

struct ResolvedSingleMoleculePattern {
    ::NFcore::MoleculeType *mt = nullptr;
    std::vector<std::pair<int, ResolvedComponentConstraint>> nonclass; // (index, constraint)
    std::vector<ResolvedClassGroup> classes;
};

static int constraint_state_key(const ResolvedComponentConstraint &c) {
    return c.has_state ? c.state : -1;
}

static bool constraint_less(const ResolvedComponentConstraint &a,
                            const ResolvedComponentConstraint &b) {
    int ka = constraint_state_key(a);
    int kb = constraint_state_key(b);
    if (ka != kb)
        return ka < kb;
    return static_cast<int>(a.bond) < static_cast<int>(b.bond);
}

static ResolvedSingleMoleculePattern
resolve_exact_single_molecule_pattern(::NFcore::System *system, const std::string &pattern) {
    ParsedSpeciesPattern parsed = parse_species_pattern(pattern);
    if (parsed.molecules.size() != 1) {
        throw std::runtime_error(
            "NfsimSimulator species APIs currently support exact single-molecule patterns only: '" +
            pattern + "'");
    }

    const ParsedMoleculePattern &mol = parsed.molecules.front();
    if (mol.name.empty()) {
        throw std::runtime_error("invalid BNGL pattern: molecule without name");
    }

    ::NFcore::MoleculeType *mt = system->getMoleculeTypeByName(mol.name);
    if (!mt) {
        throw std::runtime_error("NfsimSimulator species API: unknown MoleculeType '" + mol.name +
                                 "'");
    }

    const int n_components = mt->getNumOfComponents();
    std::map<int, std::vector<int>> class_to_indices; // class id -> sorted indices
    for (int c = 0; c < n_components; ++c) {
        int cid = mt->getEquivalenceClassNumber(c);
        if (cid >= 0) {
            class_to_indices[cid].push_back(c);
        }
    }
    for (auto &kv : class_to_indices) {
        std::sort(kv.second.begin(), kv.second.end());
    }

    std::vector<bool> claimed(n_components, false);
    std::map<int, std::vector<ResolvedComponentConstraint>> class_constraints;
    std::vector<std::pair<int, ResolvedComponentConstraint>> nonclass_constraints;

    for (const ParsedComponentPattern &pc : mol.components) {
        if (pc.bond == BondConstraint::Explicit || pc.bond == BondConstraint::Bound ||
            pc.bond == BondConstraint::Any) {
            throw std::runtime_error(
                "NfsimSimulator species APIs currently support unbound single-molecule species "
                "patterns only: '" +
                pattern + "'");
        }

        ResolvedComponentConstraint con;
        con.bond = pc.bond;

        // Try direct lookup first. This handles non-class names (unique components)
        // and also lets users disambiguate symmetric sites explicitly via NFsim's
        // internal "r1"/"r2" suffixed names if they want to.
        int direct_index = find_component_index(mt, pc.name);
        if (direct_index >= 0) {
            if (claimed[direct_index]) {
                throw std::runtime_error("invalid BNGL pattern: duplicate component '" + pc.name +
                                         "'");
            }
            if (pc.state.has_value()) {
                int state_value = find_state_value(mt, direct_index, *pc.state);
                if (state_value < 0) {
                    throw std::runtime_error("NfsimSimulator species API: unknown state '" +
                                             *pc.state + "' for component '" + pc.name +
                                             "' on MoleculeType '" + mol.name + "'");
                }
                con.has_state = true;
                con.state = state_value;
            }
            claimed[direct_index] = true;
            int cid = mt->getEquivalenceClassNumber(direct_index);
            if (cid >= 0) {
                class_constraints[cid].push_back(con);
            } else {
                nonclass_constraints.emplace_back(direct_index, con);
            }
            continue;
        }

        // Direct lookup failed. Maybe `pc.name` is the bare class-original
        // name (e.g., "r") whose physical sites were renamed to "r1","r2"
        // by NFsim's loader. Take the next free slot from the class.
        if (mt->isEquivalentComponent(pc.name)) {
            int cid = mt->getEquivalencyClassNumber(pc.name);
            auto it = class_to_indices.find(cid);
            if (it == class_to_indices.end() || it->second.empty()) {
                throw std::runtime_error("NfsimSimulator species API: unknown component '" +
                                         pc.name + "' on MoleculeType '" + mol.name + "'");
            }
            int slot_idx = -1;
            for (int idx : it->second) {
                if (!claimed[idx]) {
                    slot_idx = idx;
                    break;
                }
            }
            if (slot_idx < 0) {
                throw std::runtime_error("invalid BNGL pattern: duplicate component '" + pc.name +
                                         "'");
            }
            if (pc.state.has_value()) {
                int state_value = find_state_value(mt, slot_idx, *pc.state);
                if (state_value < 0) {
                    throw std::runtime_error("NfsimSimulator species API: unknown state '" +
                                             *pc.state + "' for component '" + pc.name +
                                             "' on MoleculeType '" + mol.name + "'");
                }
                con.has_state = true;
                con.state = state_value;
            }
            claimed[slot_idx] = true;
            class_constraints[cid].push_back(con);
            continue;
        }

        throw std::runtime_error("NfsimSimulator species API: unknown component '" + pc.name +
                                 "' on MoleculeType '" + mol.name + "'");
    }

    for (int c = 0; c < n_components; ++c) {
        if (!claimed[c]) {
            throw std::runtime_error(
                "NfsimSimulator species APIs require exact single-molecule species patterns with "
                "every component listed: '" +
                pattern + "'");
        }
    }

    auto possible_states = mt->getPossibleCompStates();
    for (auto &kv : nonclass_constraints) {
        int idx = kv.first;
        if (!possible_states[idx].empty() && !kv.second.has_state) {
            throw std::runtime_error(
                "NfsimSimulator species APIs require explicit states for stateful components "
                "in exact mutation patterns: '" +
                pattern + "'");
        }
    }
    for (auto &kv : class_constraints) {
        const auto &indices = class_to_indices[kv.first];
        if (indices.empty())
            continue;
        bool stateful = !possible_states[indices.front()].empty();
        if (stateful) {
            for (const auto &con : kv.second) {
                if (!con.has_state) {
                    throw std::runtime_error(
                        "NfsimSimulator species APIs require explicit states for stateful "
                        "components in exact mutation patterns: '" +
                        pattern + "'");
                }
            }
        }
    }

    ResolvedSingleMoleculePattern resolved;
    resolved.mt = mt;
    resolved.nonclass = std::move(nonclass_constraints);
    for (auto &kv : class_constraints) {
        ResolvedClassGroup group;
        group.indices = class_to_indices[kv.first];
        group.sorted_constraints = std::move(kv.second);
        std::sort(group.sorted_constraints.begin(), group.sorted_constraints.end(),
                  constraint_less);
        group.stateful =
            !group.sorted_constraints.empty() && group.sorted_constraints.front().has_state;
        resolved.classes.push_back(std::move(group));
    }
    return resolved;
}

static bool molecule_matches(const ::NFcore::Molecule *mol,
                             const ResolvedSingleMoleculePattern &pattern) {
    if (!mol || !mol->isAlive() || mol->getMoleculeType() != pattern.mt) {
        return false;
    }

    for (const auto &kv : pattern.nonclass) {
        int idx = kv.first;
        const ResolvedComponentConstraint &con = kv.second;
        if (con.has_state && mol->getComponentState(idx) != con.state) {
            return false;
        }
        if (con.bond == BondConstraint::Unbound && !mol->isBindingSiteOpen(idx)) {
            return false;
        }
    }

    for (const ResolvedClassGroup &group : pattern.classes) {
        // Bond check is per-index; only Unbound is supported and we apply it
        // uniformly across class members (same reasoning as non-class path).
        for (int idx : group.indices) {
            if (!group.sorted_constraints.empty() &&
                group.sorted_constraints.front().bond == BondConstraint::Unbound &&
                !mol->isBindingSiteOpen(idx)) {
                return false;
            }
        }
        if (!group.stateful) {
            continue;
        }
        std::vector<int> actual;
        actual.reserve(group.indices.size());
        for (int idx : group.indices) {
            actual.push_back(mol->getComponentState(idx));
        }
        std::sort(actual.begin(), actual.end());
        if (actual.size() != group.sorted_constraints.size()) {
            return false;
        }
        for (size_t i = 0; i < actual.size(); ++i) {
            if (actual[i] != group.sorted_constraints[i].state) {
                return false;
            }
        }
    }
    return true;
}

static std::vector<::NFcore::Molecule *>
matching_molecules(const ResolvedSingleMoleculePattern &pattern) {
    std::vector<::NFcore::Molecule *> matches;
    const int n_molecules = pattern.mt->getMoleculeCount();
    for (int i = 0; i < n_molecules; ++i) {
        ::NFcore::Molecule *mol = pattern.mt->getMolecule(i);
        if (molecule_matches(mol, pattern)) {
            matches.push_back(mol);
        }
    }
    return matches;
}

// ─── BNG XML <Parameter> table for ExprTk-driven override propagation ────────
//
// NFsim's loader records only the precomputed `value=` for each `<Parameter>`,
// dropping the `expr=` attribute. That makes setParameter() a flat write —
// dependents like `LT = LT_conc_M*NA*V_sim` or `_rateLawN = kf1*(1-use_excess)`
// stay pinned to their XML-time values when callers move a base parameter.
//
// XmlParamTable closes the gap: it parses every `<Parameter>` tag once, holds
// the BNG-emitted expression strings, and re-evaluates the whole table through
// bngsim's ExprTk evaluator whenever overrides change. The new values are
// pushed into NFsim's paramMap and updateSystemWithNewParameters() cascades
// them through global / composite / local functions and reaction base rates.
//
// Extract a quoted attribute value: name="..." (or name='...').
// BNG2.pl uses double quotes throughout; supporting both keeps this
// resilient to any future XML-emitter tweaks.
//
// Returns the inner string and (out-param) the [open_quote_index,
// close_quote_index] range so callers that want to splice replacement
// text into the original line can do so without re-scanning.
static std::optional<std::string> extract_xml_attr(const std::string &line, const std::string &attr,
                                                   std::size_t *out_value_begin = nullptr,
                                                   std::size_t *out_value_end = nullptr) {
    const std::string needle = attr + "=";
    size_t pos = 0;
    while ((pos = line.find(needle, pos)) != std::string::npos) {
        if (pos > 0) {
            char prev = line[pos - 1];
            if (std::isalnum(static_cast<unsigned char>(prev)) || prev == '_') {
                pos += needle.size();
                continue;
            }
        }
        size_t q = pos + needle.size();
        if (q >= line.size() || (line[q] != '"' && line[q] != '\'')) {
            return std::nullopt;
        }
        char quote = line[q];
        size_t end = line.find(quote, q + 1);
        if (end == std::string::npos) {
            return std::nullopt;
        }
        if (out_value_begin) {
            *out_value_begin = q + 1;
        }
        if (out_value_end) {
            *out_value_end = end;
        }
        return line.substr(q + 1, end - q - 1);
    }
    return std::nullopt;
}

struct XmlParam {
    std::string name;
    std::string expr;
    double xml_value = 0.0;
};

class XmlParamTable {
  public:
    bool loaded = false;
    std::vector<XmlParam> params;                       // declaration order
    std::unordered_map<std::string, std::size_t> index; // name → params slot

    void load(const std::string &xml_path) {
        if (loaded) {
            return;
        }
        std::ifstream in(xml_path);
        if (!in) {
            throw std::runtime_error("XmlParamTable: cannot open XML file '" + xml_path + "'");
        }
        std::string line;
        // BNG2.pl emits one <Parameter id="..." [type="..."] value="..." [expr="..."]/> per line
        // inside <ListOfParameters>...</ListOfParameters>. We scan linewise and stop at the
        // closing tag — anything later in the XML (functions, observables, reactions) does
        // not concern this table.
        bool in_list = false;
        while (std::getline(in, line)) {
            if (!in_list) {
                if (line.find("<ListOfParameters>") != std::string::npos) {
                    in_list = true;
                }
                continue;
            }
            if (line.find("</ListOfParameters>") != std::string::npos) {
                break;
            }
            auto trimmed = trim_ascii(line);
            if (trimmed.rfind("<Parameter", 0) != 0) {
                continue;
            }
            auto id = extract_xml_attr(trimmed, "id");
            if (!id) {
                continue;
            }
            XmlParam p;
            p.name = *id;
            if (auto v = extract_xml_attr(trimmed, "value")) {
                try {
                    p.xml_value = std::stod(*v);
                } catch (const std::exception &) {
                    p.xml_value = 0.0;
                }
            }
            if (auto e = extract_xml_attr(trimmed, "expr")) {
                p.expr = *e;
            }
            index[p.name] = params.size();
            params.push_back(std::move(p));
        }
        loaded = true;
    }
};

// ─── BNG XML <ListOfFunctions> → expression (function) output columns ────────
//
// NFsim evaluates BNG functions internally, but the embedded System exposes no
// public enumerator for its function tables, so a bngsim Result built from a
// session reports observables only — model outputs defined in a `begin
// functions` block (e.g. `pre1_dose() = alpha*Clusters()/f`) go missing.
//
// We close the gap on the bngsim side, with no vendored-NFsim change: read the
// function names from the XML's <ListOfFunctions> block and resolve each
// against the System. NFsim splits the block in two (parseFuncXML.cpp): a
// function that references only observables/parameters is a GlobalFunction; one
// that references another function is a CompositeFunction. Both are reachable —
// getGlobalFunctionByName() + FuncFactory::Eval for the former,
// getCompositeFunctionByName() + evaluateOn() for the latter. (Standalone
// NFsim's -ogf output writer only walks global functions, so composite outputs
// such as pre1_dose are absent from its .gdat; resolving both here is a strict
// superset.) Local (molecule-scoped) functions have no scalar value and are
// skipped.

// Extract the function ids declared in the XML's <ListOfFunctions> block.
// BNG2.pl emits one <Function id="..." .../> opening tag per line.
static std::vector<std::string> extract_function_names(const std::string &xml_path) {
    std::vector<std::string> names;
    std::ifstream in(xml_path);
    if (!in) {
        return names;
    }
    std::string line;
    bool in_list = false;
    while (std::getline(in, line)) {
        if (!in_list) {
            if (line.find("<ListOfFunctions>") != std::string::npos) {
                in_list = true;
            }
            continue;
        }
        if (line.find("</ListOfFunctions>") != std::string::npos) {
            break;
        }
        auto trimmed = trim_ascii(line);
        if (trimmed.rfind("<Function", 0) != 0) {
            continue;
        }
        if (auto id = extract_xml_attr(trimmed, "id")) {
            names.push_back(*id);
        }
    }
    return names;
}

// One resolved output function — either a global or a composite. Exactly one
// pointer is non-null; the other stays null.
struct NfsimOutputFunction {
    ::NFcore::GlobalFunction *global = nullptr;
    ::NFcore::CompositeFunction *composite = nullptr;
};

// A resolved set of functions to report as Result expression columns.
struct NfsimFunctionSet {
    std::vector<std::string> names;
    std::vector<NfsimOutputFunction> funcs;
};

// Evaluate one resolved output function against the System's current state.
// Globals go through the same FuncFactory::Eval path NFsim's own output writer
// uses; composites through CompositeFunction::evaluateOn, which itself
// re-evaluates the global functions it references. File-backed (tfun) globals
// are refreshed first.
static double eval_output_function(const NfsimOutputFunction &f) {
    if (f.global != nullptr) {
        if (f.global->fileFunc) {
            f.global->fileUpdate();
        }
        return ::NFcore::FuncFactory::Eval(f.global->p);
    }
    // Composite. A scope-free evaluateOn() is valid only when the composite
    // has neither a local-function dependency (molecule-scoped) nor a
    // reactant-count dependency (reactant_N() in the rate law). Both make the
    // function meaningless without a reaction context and cause evaluateOn() to
    // throw on a NULL scope/reactant-count; resolve_output_functions() probes
    // for that and only keeps composites that evaluate cleanly here.
    return f.composite->evaluateOn(nullptr, nullptr, nullptr, 0);
}

// Resolve XML-declared function names against the System. A name resolves to a
// global, or to a composite (kept only if a scope-free evaluation succeeds —
// composites with local-function dependencies have no scalar value), or is
// skipped (local function or absent).
static NfsimFunctionSet resolve_output_functions(::NFcore::System *system,
                                                 const std::vector<std::string> &xml_names) {
    NfsimFunctionSet fs;
    for (const auto &name : xml_names) {
        NfsimOutputFunction f;
        if (::NFcore::GlobalFunction *gf = system->getGlobalFunctionByName(name)) {
            f.global = gf;
        } else if (::NFcore::CompositeFunction *cf = system->getCompositeFunctionByName(name)) {
            // Probe once: a composite that depends on local (molecule-scoped)
            // functions or on reactant counts (reactant_N()) throws on a
            // scope-free evaluateOn(); skip those — they have no scalar value
            // outside a reaction context. NFsim writes a "Time to quit"
            // cout/cerr line before throwing in the local-function branch, so
            // suppress streams during the probe to keep initialize() output
            // clean for every well-formed NF model.
            try {
                StreamSuppressor suppress;
                cf->evaluateOn(nullptr, nullptr, nullptr, 0);
            } catch (...) {
                continue;
            }
            f.composite = cf;
        } else {
            continue;
        }
        fs.names.push_back(name);
        fs.funcs.push_back(f);
    }
    return fs;
}

// Evaluate every resolved function into `out` (sized to `funcs`).
static void eval_function_set(const std::vector<NfsimOutputFunction> &funcs,
                              std::vector<double> &out) {
    for (std::size_t k = 0; k < funcs.size(); ++k) {
        out[k] = eval_output_function(funcs[k]);
    }
}

// Re-evaluate every parameter through ExprTk with `overrides` applied.
//
// Returns name→value for every parameter in the table (overrides included).
// Evaluation walks the table in declaration order — BNG2.pl emits parameters
// such that every reference is to a previously-declared name, so a single
// pass produces a fixed point.
//
// Throws std::runtime_error if any parameter expression fails to compile or
// evaluate; that is intentionally loud because silently leaving downstream
// parameters at stale values is exactly the bug we are fixing.
static std::unordered_map<std::string, double>
evaluate_param_table_with_overrides(const XmlParamTable &table,
                                    const std::unordered_map<std::string, double> &overrides) {
    std::unordered_map<std::string, double> out;
    if (table.params.empty()) {
        return out;
    }

    ExprTkEvaluator evaluator;
    std::vector<double> slots(table.params.size());

    for (size_t i = 0; i < table.params.size(); ++i) {
        slots[i] = table.params[i].xml_value;
        try {
            evaluator.define_variable(table.params[i].name, &slots[i]);
        } catch (const std::exception &e) {
            throw std::runtime_error("XmlParamTable: cannot register parameter '" +
                                     table.params[i].name + "' with ExprTk evaluator (" + e.what() +
                                     "). Note: BNG built-in names like "
                                     "'time' clash with ExprTk's built-in functions.");
        }
    }

    std::vector<int> compiled(table.params.size(), -1);
    for (size_t i = 0; i < table.params.size(); ++i) {
        const auto &p = table.params[i];
        if (p.expr.empty()) {
            continue;
        }
        try {
            compiled[i] = evaluator.compile(p.expr);
        } catch (const std::exception &e) {
            throw std::runtime_error("XmlParamTable: failed to compile expression for parameter '" +
                                     p.name + "' (expr='" + p.expr + "'): " + e.what());
        }
    }

    for (size_t i = 0; i < table.params.size(); ++i) {
        const auto &p = table.params[i];
        auto it = overrides.find(p.name);
        if (it != overrides.end()) {
            slots[i] = it->second;
        } else if (compiled[i] >= 0) {
            slots[i] = evaluator.evaluate(compiled[i]);
        }
        out[p.name] = slots[i];
    }
    return out;
}

// Write a per-process unique temp XML alongside `tmp_dir`. Caller owns
// the returned path and must remove the file when done.
//
// Why a temp file (and not in-memory string): `NFinput::initializeFromXML`
// takes a path. The cost (one ~50 KB write + one ~50 KB read) is
// negligible next to NFsim parsing.
static std::filesystem::path make_unique_tmp_xml_path() {
    namespace fs = std::filesystem;
    static std::atomic<uint64_t> counter{0};
    const auto pid = static_cast<long>(::getpid());
    const auto n = counter.fetch_add(1, std::memory_order_relaxed);
    fs::path tmp_dir = fs::temp_directory_path();
    return tmp_dir / ("bngsim_nfsim_" + std::to_string(pid) + "_" + std::to_string(n) + ".xml");
}

// Format an override-resolved value with full IEEE 754 precision so the
// rewritten XML round-trips exactly to the same bit pattern. NFsim parses
// `value=` via convertToDouble → strtod, which accepts `%.17g` output.
static std::string format_override_value(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return std::string(buf);
}

// Splice a new `value="..."` attribute payload into a `<Parameter ...>` line,
// leaving everything else (including any `expr="..."` attribute) untouched.
// Returns the line unchanged if `value=` is missing or malformed.
static std::string rewrite_parameter_value_attr(const std::string &line, double new_value) {
    std::size_t value_begin = 0;
    std::size_t value_end = 0;
    auto current = extract_xml_attr(line, "value", &value_begin, &value_end);
    if (!current) {
        return line;
    }
    return line.substr(0, value_begin) + format_override_value(new_value) + line.substr(value_end);
}

// Write a copy of `src_xml_path` to a fresh temp file, replacing the
// `value="..."` attribute on every `<Parameter id="X" .../>` line whose id
// appears in `values`. All other content (including `<Species
// concentration="X">` references and `<RateConstant value="_rateLawN"/>`
// pointers) is preserved verbatim — NFsim resolves those through the
// parameter map, so rewriting `<Parameter value=>` cascades automatically.
//
// Throws std::runtime_error on I/O failure. Caller must remove the
// returned path when done.
static std::filesystem::path
write_param_overridden_xml(const std::string &src_xml_path,
                           const std::unordered_map<std::string, double> &values) {
    namespace fs = std::filesystem;
    std::ifstream in(src_xml_path);
    if (!in) {
        throw std::runtime_error("write_param_overridden_xml: cannot open source XML '" +
                                 src_xml_path + "'");
    }

    fs::path tmp_path = make_unique_tmp_xml_path();
    std::ofstream out(tmp_path);
    if (!out) {
        throw std::runtime_error("write_param_overridden_xml: cannot open temp XML '" +
                                 tmp_path.string() + "'");
    }

    bool in_param_list = false;
    std::string line;
    while (std::getline(in, line)) {
        if (!in_param_list) {
            if (line.find("<ListOfParameters>") != std::string::npos) {
                in_param_list = true;
            }
        } else if (line.find("</ListOfParameters>") != std::string::npos) {
            in_param_list = false;
        } else if (trim_ascii(line).rfind("<Parameter", 0) == 0) {
            if (auto id = extract_xml_attr(line, "id")) {
                auto it = values.find(*id);
                if (it != values.end()) {
                    line = rewrite_parameter_value_attr(line, it->second);
                }
            }
        }
        out << line << '\n';
    }
    out.close();
    if (!out) {
        std::error_code ec;
        fs::remove(tmp_path, ec);
        throw std::runtime_error("write_param_overridden_xml: write failure on temp XML '" +
                                 tmp_path.string() + "'");
    }
    return tmp_path;
}

// RAII wrapper that removes the temp XML on scope exit. Best-effort: any
// filesystem error during cleanup is swallowed so it can't mask a more
// important error from the surrounding parse path.
class TempXmlGuard {
  public:
    explicit TempXmlGuard(std::filesystem::path path) : path_(std::move(path)) {}
    ~TempXmlGuard() {
        if (!path_.empty()) {
            std::error_code ec;
            std::filesystem::remove(path_, ec);
        }
    }
    TempXmlGuard(const TempXmlGuard &) = delete;
    TempXmlGuard &operator=(const TempXmlGuard &) = delete;
    const std::filesystem::path &path() const { return path_; }

  private:
    std::filesystem::path path_;
};

// ─── NfsimSimulator::Impl ────────────────────────────────────────────────────

struct NfsimSimulator::Impl {
    std::string xml_path;
    int molecule_limit = 2147483647; // INT_MAX — no artificial limit
    bool connectivity_flag = false;
    bool nfsim_v1143_compat = false;
    // bngsim defaults -bscb ON (correctness for BLBR-style models). NFsim CLI
    // defaults it off; bngsim deliberately differs.
    bool block_same_complex_binding = true;
    int traversal_limit = -1; // -1 = ReactionClass::NO_LIMIT (auto)

    // Parameter overrides: applied via setParameter() before each run
    std::unordered_map<std::string, double> param_overrides;

    // BNG XML parameter table — lazily loaded the first time we need to
    // propagate an override to dependents. Populated from the same XML the
    // NFsim System is built from, but kept separate so it remains accessible
    // before initialize() and survives session destroy/restart cycles.
    XmlParamTable param_table;

    // Session state: owned System that persists between calls
    ::NFcore::System *session_system = nullptr;
    int session_n_obs = 0;
    std::vector<std::string> session_obs_names;
    // Global functions reported as Result expression columns. Resolved at
    // initialize() against session_system; the pointers it owns stay valid
    // until the session System is destroyed or recreated.
    std::vector<std::string> session_fn_names;
    std::vector<NfsimOutputFunction> session_fn_funcs;
    double session_logical_time = 0.0;
    // Named concentration snapshots, keyed by label ("" = default/unlabeled
    // slot). Each label owns its own SystemSnapshot, so multiple named states
    // coexist — true multi-slot parity with the network-based Model (issue #11).
    // This bypasses NFcore::System's single internal savedSnapshot. Cleared
    // whenever the session System is (re)created or destroyed, since the
    // captured molecules belong to that System.
    std::unordered_map<std::string, std::unique_ptr<::NFcore::SystemSnapshot>> session_snapshots;

    Impl(const std::string &path) : xml_path(path) {}

    ~Impl() {
        if (session_system) {
            delete session_system;
            session_system = nullptr;
        }
        session_logical_time = 0.0;
    }

    // Parse XML and set up a fresh system (shared by run() and initialize())
    ::NFcore::System *create_system() {
        StreamSuppressor suppress;
        // initializeFromXML may modify suggestedTraversalLimit; seeding it
        // with -1 (NO_LIMIT) matches NFsim CLI defaults.
        int suggestedTraversalLimit = -1;
        std::filesystem::path xml_file = std::filesystem::absolute(xml_path);

        // If overrides are pending, bake them into a temp XML before NFsim
        // parses. NFsim's loader resolves `<Species concentration="X">` and
        // `<RateConstant value="_rateLawN"/>` against the parameter map at
        // parse time, so once we have written propagated values into
        // `<Parameter value="...">`, downstream agents and rate constants
        // pick up the overridden namespace automatically. Without this,
        // setParameter calls before initialize() left seed-species agent
        // counts pinned to the XML-time value (issue #29).
        std::optional<TempXmlGuard> temp_guard;
        std::optional<TempXmlGuard> seed_round_guard;
        std::filesystem::path xml_to_load = xml_file;
        std::unordered_map<std::string, double> resolved_param_values;
        if (!param_overrides.empty()) {
            param_table.load(xml_path);
            if (!param_table.params.empty()) {
                resolved_param_values =
                    evaluate_param_table_with_overrides(param_table, param_overrides);
                temp_guard.emplace(
                    write_param_overridden_xml(xml_file.string(), resolved_param_values));
                xml_to_load = temp_guard->path();
            }
        }

        // Round fractional seed-species concentrations to integer counts at the
        // bngsim handoff so the vendored NFsim loader's truncating (int) cast
        // becomes a no-op (GH #51). resolved_param_values supplies override-aware
        // values for parameter-id concentrations; an empty overlay falls back to
        // the XML's own value= attrs. No-op (and no temp file) when every seed
        // concentration is already integer.
        if (auto rounded = bngsim::seedround::write_seed_rounded_xml(xml_to_load.string(),
                                                                     resolved_param_values)) {
            seed_round_guard.emplace(*rounded);
            xml_to_load = seed_round_guard->path();
        }

        // Resolve relative TFUN/file dependencies against the XML directory.
        // NFsim opens external files using process CWD, not XML-relative paths.
        // We chdir to the *original* XML's directory even when loading a
        // rewritten copy from /tmp, so any TFUN-by-relative-path references
        // continue to resolve.
        std::filesystem::path previous_cwd;
        bool changed_cwd = false;
        try {
            std::filesystem::path xml_dir = xml_file.parent_path();
            if (!xml_dir.empty()) {
                previous_cwd = std::filesystem::current_path();
                std::filesystem::current_path(xml_dir);
                changed_cwd = true;
            }
        } catch (...) {
            // Best effort: if cwd switching fails, keep original cwd behavior.
            changed_cwd = false;
        }

        // Pass through the same-complex-binding-block flag (NFsim ``-bscb``).
        // When ``false``, NFsim still auto-enables complex tracking if the
        // model declares any Species-typed observable.
        auto *system = NFinput::initializeFromXML(
            xml_to_load.string(),
            /*blockSameComplexBinding=*/block_same_complex_binding, molecule_limit,
            /*verbose=*/false, suggestedTraversalLimit,
            /*evaluateComplexScopedLocalFunctions=*/true,
            /*connectivityFlag=*/connectivity_flag);

        if (changed_cwd) {
            try {
                std::filesystem::current_path(previous_cwd);
            } catch (...) {
                // Ignore restoration failure; caller will still receive load result.
            }
        }

        if (!system) {
            std::string msg = "NfsimSimulator: Failed to load XML file: " + xml_path;
            std::string details = summarize_nfsim_log(suppress.captured_output());
            if (!details.empty()) {
                msg += " | NFsim log: " + details;
            }
            throw std::runtime_error(msg);
        }
        // -utl semantics: explicit user value overrides NFsim's auto-suggested
        // limit. Negative traversal_limit means "use the auto-suggested limit".
        const int utl = (traversal_limit < 0) ? suggestedTraversalLimit : traversal_limit;
        system->setUniversalTraversalLimit(utl);
        system->setNFsimV1143Compatibility(nfsim_v1143_compat);
        return system;
    }

    // Prepare system for simulation (shared setup)
    void prepare_system(::NFcore::System *system, uint64_t seed) {
        StreamSuppressor suppress;
        try {
            system->seedRNG(static_cast<unsigned long>(seed));
            // Temporary parity bridge while vendored runtime still has global
            // NFutil RNG callsites (e.g., ReactantList::pickRandom).
            NFutil::SEED_RANDOM(static_cast<unsigned long>(seed));
            system->registerOutputFileLocation(bngsim::null_device);
            system->outputAllObservableNames();
            system->prepareForSimulation();

            // Apply parameter overrides after prepareForSimulation. Pre-init
            // overrides are propagated through the BNG XML expression graph so
            // dependents (e.g., LT, _rateLawN, use_excess) re-evaluate against
            // the new namespace.
            if (!param_overrides.empty()) {
                push_overrides_to_system(system);
            }
        } catch (const std::exception &e) {
            // NFsim signals post-parse setup failures (e.g. unsupported
            // functional-rate constructs) by printing a diagnostic to
            // cout/cerr and then throwing a bare "Quitting". StreamSuppressor
            // has captured that diagnostic; surface it the same way
            // create_system() relays NFsim's log on XML-load failure, instead
            // of letting the actionless "Quitting" propagate alone (issue #63).
            std::string msg = std::string("NFsim setup failed: ") + e.what();
            std::string details = summarize_nfsim_log(suppress.captured_output());
            if (!details.empty()) {
                msg += " | NFsim log: " + details;
            }
            throw std::runtime_error(msg);
        }
    }

    // Push the current param_overrides to a live NFsim System, propagating
    // re-evaluated dependent parameters and refreshing every reaction rate.
    void push_overrides_to_system(::NFcore::System *system) {
        param_table.load(xml_path);
        if (param_table.params.empty()) {
            // No XML parameters parsed; fall back to the legacy flat write so
            // we still respect explicit user overrides on names NFsim knows.
            for (const auto &[name, value] : param_overrides) {
                system->setParameter(name, value);
            }
            system->updateSystemWithNewParameters();
            return;
        }

        auto values = evaluate_param_table_with_overrides(param_table, param_overrides);
        for (const auto &[name, value] : values) {
            system->setParameter(name, value);
        }
        // Forward any user override whose name is not in the XML param table
        // (defensive: shouldn't happen for well-formed BNG XML, but preserves
        // pre-fix behavior for edge cases).
        for (const auto &[name, value] : param_overrides) {
            if (param_table.index.find(name) == param_table.index.end()) {
                system->setParameter(name, value);
            }
        }
        system->updateSystemWithNewParameters();
    }

    // Cache observable names from system
    void cache_obs_names(::NFcore::System *system) {
        session_n_obs = system->getNumOfObsForOutput();
        session_obs_names.resize(session_n_obs);
        for (int i = 0; i < session_n_obs; ++i) {
            session_obs_names[i] = system->getObsForOutput(i)->getName();
        }
    }

    // Resolve the XML-declared global functions against `system` and cache
    // them for this session's simulate() calls.
    void cache_function_set(::NFcore::System *system) {
        auto fs = resolve_output_functions(system, extract_function_names(xml_path));
        session_fn_names = std::move(fs.names);
        session_fn_funcs = std::move(fs.funcs);
    }

    void require_session() const {
        if (!session_system) {
            throw std::runtime_error("NfsimSimulator: No active session. Call initialize() first.");
        }
    }
};

// ─── Constructor / Destructor ────────────────────────────────────────────────

NfsimSimulator::NfsimSimulator(const std::string &xml_path)
    : impl_(std::make_unique<Impl>(xml_path)) {}

NfsimSimulator::~NfsimSimulator() = default;

// ─── Configuration ───────────────────────────────────────────────────────────

void NfsimSimulator::set_param(const std::string &name, double value) {
    impl_->param_overrides[name] = value;
    // When a session is already active, push the override (and propagated
    // dependents) into the live NFsim System immediately. Without this the
    // post-init write was silently dropped — see issue #20.
    if (impl_->session_system) {
        StreamSuppressor suppress;
        impl_->push_overrides_to_system(impl_->session_system);
        // Reaction propensities just changed — drop NFsim's stepTo cache so
        // the next simulate() call re-samples next-firing times.
        impl_->session_system->invalidateStepToCache();
    }
}

void NfsimSimulator::clear_param_overrides() {
    impl_->param_overrides.clear();
    // Mirror set_param: a live session gets its parameters reset to their
    // XML-time values via re-eval-with-empty-overrides.
    if (impl_->session_system) {
        StreamSuppressor suppress;
        impl_->push_overrides_to_system(impl_->session_system);
        impl_->session_system->invalidateStepToCache();
    }
}

double
NfsimSimulator::evaluate_expression(const std::string &expr,
                                    const std::unordered_map<std::string, double> &extra) const {
    impl_->param_table.load(impl_->xml_path);

    // Layer extra overrides on top of the simulator's persistent overrides.
    // Caller-provided values win on collision because they reflect the
    // immediate intent of this evaluate() call.
    std::unordered_map<std::string, double> overrides = impl_->param_overrides;
    for (const auto &[name, value] : extra) {
        overrides[name] = value;
    }

    auto values = evaluate_param_table_with_overrides(impl_->param_table, overrides);

    ExprTkEvaluator evaluator;
    std::vector<double> slots;
    slots.reserve(values.size());
    // Stable iteration order so pointers stay valid for the life of the call.
    for (const auto &p : impl_->param_table.params) {
        slots.push_back(values.at(p.name));
    }
    for (size_t i = 0; i < impl_->param_table.params.size(); ++i) {
        evaluator.define_variable(impl_->param_table.params[i].name, &slots[i]);
    }
    int id;
    try {
        id = evaluator.compile(expr);
    } catch (const std::exception &e) {
        throw std::runtime_error("NfsimSimulator::evaluate_expression: failed to compile '" + expr +
                                 "': " + e.what());
    }
    return evaluator.evaluate(id);
}

void NfsimSimulator::set_molecule_limit(int limit) { impl_->molecule_limit = limit; }

void NfsimSimulator::set_connectivity(bool enabled) { impl_->connectivity_flag = enabled; }

void NfsimSimulator::set_nfsim_v1143_compat(bool enabled) {
    impl_->nfsim_v1143_compat = enabled;
    if (impl_->session_system) {
        impl_->session_system->setNFsimV1143Compatibility(enabled);
    }
}

void NfsimSimulator::set_block_same_complex_binding(bool enabled) {
    impl_->block_same_complex_binding = enabled;
}

void NfsimSimulator::set_traversal_limit(int limit) { impl_->traversal_limit = limit; }

const std::string &NfsimSimulator::xml_path() const { return impl_->xml_path; }

// ─── run() — stateless simulation (re-parses XML each call) ──────────────────

Result NfsimSimulator::run(const TimeSpec &times, uint64_t seed, double timeout_seconds) {
    auto *system = impl_->create_system();
    impl_->prepare_system(system, seed);

    // Wall-clock budget. Checked between each stepTo() because NFsim's
    // sampler is opaque from the bngsim side — the smallest unit we can
    // interrupt is one output-point hop.
    WallClockBudget budget(timeout_seconds);

    // Read observable names
    const int n_obs = system->getNumOfObsForOutput();
    std::vector<std::string> obs_names(n_obs);
    for (int i = 0; i < n_obs; ++i) {
        obs_names[i] = system->getObsForOutput(i)->getName();
    }

    // Compute output time points. output_times() returns the explicit
    // sample_times when the caller supplied them (sorted ascending, with
    // sample_times[0] == t_start by the TimeSpec contract), otherwise the
    // uniform t_start..t_end / n_points grid. NFsim's stepTo() targets each
    // instant directly, so honoring an arbitrary, non-uniform list is just a
    // matter of iterating the resolved vector (GH #169). This mirrors BNG2.pl's
    // simulate_nf sample_times branch (BNGAction.pm).
    std::vector<double> t_out = times.output_times();
    const int n_points = static_cast<int>(t_out.size());

    // Allocate Result (0 species columns — NFsim only has observables)
    Result result;
    result.allocate(n_points, /*n_species=*/0, n_obs);
    result.set_observable_names(obs_names);

    // Global functions → Result expression columns.
    NfsimFunctionSet fn_set =
        resolve_output_functions(system, extract_function_names(impl_->xml_path));
    result.set_expression_names(fn_set.names);

    // Observable + function value buffers
    std::vector<double> obs_buf(n_obs);
    std::vector<double> fn_buf(fn_set.names.size());

    // Record initial state (t=0)
    for (int j = 0; j < n_obs; ++j) {
        obs_buf[j] = static_cast<double>(system->getObsForOutput(j)->getCount());
    }
    result.record(0, t_out[0], nullptr, obs_buf.data());
    eval_function_set(fn_set.funcs, fn_buf);
    result.record_expressions(0, fn_buf.data());

    // Simulation loop using stepTo()
    try {
        StreamSuppressor suppress;
        for (int i = 1; i < n_points; ++i) {
            if (budget.active())
                budget.check();
            system->stepTo(t_out[i]);
            for (int j = 0; j < n_obs; ++j) {
                obs_buf[j] = static_cast<double>(system->getObsForOutput(j)->getCount());
            }
            result.record(i, t_out[i], nullptr, obs_buf.data());
            eval_function_set(fn_set.funcs, fn_buf);
            result.record_expressions(i, fn_buf.data());
        }
    } catch (...) {
        // Free the parsed System before propagating (no other cleanup hook
        // on this path). Catches TimeoutError as well as any NFsim-internal
        // exception that may surface here.
        delete system;
        throw;
    }

    // Solver stats
    result.solver_stats().n_steps = system->getGlobalEventCounter();
    result.solver_stats().n_rhs_evals = system->getGlobalEventCounter();

    delete system;
    return result;
}

// ─── Session API — stateful, multi-action workflows ──────────────────────────

void NfsimSimulator::initialize(uint64_t seed) {
    // Destroy any existing session
    if (impl_->session_system) {
        delete impl_->session_system;
        impl_->session_system = nullptr;
    }

    impl_->session_system = impl_->create_system();
    impl_->prepare_system(impl_->session_system, seed);
    impl_->cache_obs_names(impl_->session_system);
    impl_->cache_function_set(impl_->session_system);
    impl_->session_logical_time = 0.0;
    impl_->session_snapshots.clear();
}

void NfsimSimulator::step_to(double time) {
    impl_->require_session();
    StreamSuppressor suppress;
    impl_->session_system->stepTo(time);
    impl_->session_logical_time = time;
}

Result NfsimSimulator::simulate(double t_start, double t_end, int n_points, double timeout_seconds,
                                bool relative_time, const std::vector<double> &sample_times) {
    impl_->require_session();
    auto *system = impl_->session_system;
    const int n_obs = impl_->session_n_obs;
    const double segment_base_time = impl_->session_logical_time;

    // Wall-clock budget. Same pattern as NfsimSimulator::run() — NFsim's
    // sampler is opaque so the smallest interruptible unit is one
    // output-point hop. On overrun the session_logical_time is left at the
    // segment_base_time captured on entry; the live System has advanced an
    // unknown distance into the segment and the only safe follow-up is
    // destroy_session().
    WallClockBudget budget(timeout_seconds);

    // Resolve the output schedule. Two parallel arrays of equal length:
    //   * t_out[i]       — the user-facing label stamped onto the Result.
    //   * abs_targets[i] — the absolute NFsim time stepTo() advances to.
    // The whole segment is anchored to the session clock captured on entry so
    // labels map consistently onto the internal clock even across
    // sub-intervals with no events.
    //
    // Source of the schedule:
    //   * sample_times non-empty (GH #184): honor the caller's exact instants,
    //     treated as absolute in the caller's frame with sample_times[0] the
    //     segment start. The live clock advances by (sample_times[i] -
    //     sample_times[0]) from segment_base_time — the same offset the
    //     uniform path applies via i*dt. stepTo() targets arbitrary instants
    //     (the stateless run(TimeSpec) path already relies on this), so a
    //     non-uniform list needs no special handling beyond the loop below.
    //   * otherwise the uniform t_start..t_end / n_points grid (unchanged;
    //     byte-identical to before so the golden/parity suites are stable).
    const bool explicit_times = !sample_times.empty();
    const int n_out = explicit_times ? static_cast<int>(sample_times.size()) : n_points;

    std::vector<double> t_out(n_out);
    std::vector<double> abs_targets(n_out);
    if (explicit_times) {
        const double origin = sample_times.front();
        for (int i = 0; i < n_out; ++i) {
            const double offset = sample_times[i] - origin;
            t_out[i] = relative_time ? offset : sample_times[i];
            abs_targets[i] = segment_base_time + offset;
        }
    } else {
        const double dt = (n_out > 1) ? (t_end - t_start) / (n_out - 1) : 0.0;
        const double t_label_origin = relative_time ? 0.0 : t_start;
        for (int i = 0; i < n_out; ++i) {
            t_out[i] = t_label_origin + i * dt;
            abs_targets[i] = segment_base_time + i * dt;
        }
    }

    // Allocate Result
    Result result;
    result.allocate(n_out, /*n_species=*/0, n_obs);
    result.set_observable_names(impl_->session_obs_names);
    result.set_expression_names(impl_->session_fn_names);

    std::vector<double> obs_buf(n_obs);
    std::vector<double> fn_buf(impl_->session_fn_names.size());

    // Record initial state at the first output point (current system state)
    for (int j = 0; j < n_obs; ++j) {
        obs_buf[j] = static_cast<double>(system->getObsForOutput(j)->getCount());
    }
    result.record(0, t_out[0], nullptr, obs_buf.data());
    eval_function_set(impl_->session_fn_funcs, fn_buf);
    result.record_expressions(0, fn_buf.data());

    // stepTo() for each subsequent time point
    {
        StreamSuppressor suppress;
        for (int i = 1; i < n_out; ++i) {
            if (budget.active())
                budget.check();
            system->stepTo(abs_targets[i]);

            for (int j = 0; j < n_obs; ++j) {
                obs_buf[j] = static_cast<double>(system->getObsForOutput(j)->getCount());
            }
            result.record(i, t_out[i], nullptr, obs_buf.data());
            eval_function_set(impl_->session_fn_funcs, fn_buf);
            result.record_expressions(i, fn_buf.data());
        }
    }

    const double segment_span =
        explicit_times ? (sample_times.back() - sample_times.front()) : (t_end - t_start);
    impl_->session_logical_time = segment_base_time + segment_span;

    result.solver_stats().n_steps = system->getGlobalEventCounter();
    result.solver_stats().n_rhs_evals = system->getGlobalEventCounter();

    return result;
}

void NfsimSimulator::add_molecules(const std::string &molecule_type_name, int count) {
    impl_->require_session();
    auto *system = impl_->session_system;

    // Find the MoleculeType by name
    ::NFcore::MoleculeType *mt = system->getMoleculeTypeByName(molecule_type_name);
    if (!mt) {
        throw std::runtime_error("NfsimSimulator::add_molecules: unknown MoleculeType '" +
                                 molecule_type_name + "'");
    }

    // Add molecules in default (unbound) state using NFsim's built-in method.
    // populateWithDefaultMolecules adds to the existing count (doesn't reset).
    // Each molecule is registered with reaction lists via addMoleculeToRunningSystem.
    StreamSuppressor suppress;
    for (int i = 0; i < count; ++i) {
        ::NFcore::Molecule *mol = mt->genDefaultMolecule();
        mt->addMoleculeToRunningSystem(mol);
    }
    system->invalidateStepToCache();
}

int NfsimSimulator::get_molecule_count(const std::string &molecule_type_name) const {
    impl_->require_session();
    auto *system = impl_->session_system;

    ::NFcore::MoleculeType *mt = system->getMoleculeTypeByName(molecule_type_name);
    if (!mt) {
        throw std::runtime_error("NfsimSimulator::get_molecule_count: unknown MoleculeType '" +
                                 molecule_type_name + "'");
    }
    return mt->getMoleculeCount();
}

int NfsimSimulator::get_species_count(const std::string &pattern) const {
    impl_->require_session();
    auto resolved = resolve_exact_single_molecule_pattern(impl_->session_system, pattern);
    return static_cast<int>(matching_molecules(resolved).size());
}

void NfsimSimulator::add_species(const std::string &pattern, int count) {
    impl_->require_session();
    if (count <= 0) {
        throw std::runtime_error("NfsimSimulator::add_species: count must be positive");
    }

    auto resolved = resolve_exact_single_molecule_pattern(impl_->session_system, pattern);
    StreamSuppressor suppress;
    for (int i = 0; i < count; ++i) {
        ::NFcore::Molecule *mol = resolved.mt->genDefaultMolecule();
        for (const auto &kv : resolved.nonclass) {
            const ResolvedComponentConstraint &con = kv.second;
            if (con.has_state) {
                mol->setComponentState(kv.first, con.state);
            }
        }
        // Symmetric class members: NFsim treats any permutation of states across
        // class indices as the same species, so any consistent assignment of the
        // sorted constraints to the sorted indices realizes the requested species.
        for (const ResolvedClassGroup &group : resolved.classes) {
            if (!group.stateful) {
                continue;
            }
            for (size_t k = 0; k < group.indices.size(); ++k) {
                mol->setComponentState(group.indices[k], group.sorted_constraints[k].state);
            }
        }
        resolved.mt->addMoleculeToRunningSystem(mol);
    }
    impl_->session_system->invalidateStepToCache();
}

void NfsimSimulator::remove_species(const std::string &pattern, int count) {
    impl_->require_session();
    if (count <= 0) {
        throw std::runtime_error("NfsimSimulator::remove_species: count must be positive");
    }

    auto resolved = resolve_exact_single_molecule_pattern(impl_->session_system, pattern);
    auto matches = matching_molecules(resolved);
    if (static_cast<int>(matches.size()) < count) {
        throw std::runtime_error("NfsimSimulator::remove_species: insufficient matches for '" +
                                 pattern + "' (requested " + std::to_string(count) + ", found " +
                                 std::to_string(matches.size()) + ")");
    }

    StreamSuppressor suppress;
    for (int i = 0; i < count; ++i) {
        ::NFcore::Molecule *mol = matches[static_cast<size_t>(i)];
        resolved.mt->removeMoleculeFromRunningSystem(mol);
    }
    impl_->session_system->invalidateStepToCache();
}

void NfsimSimulator::set_species_count(const std::string &pattern, int count) {
    impl_->require_session();
    if (count < 0) {
        throw std::runtime_error("NfsimSimulator::set_species_count: count must be nonnegative");
    }

    int current = get_species_count(pattern);
    int delta = count - current;
    if (delta > 0) {
        add_species(pattern, delta);
    } else if (delta < 0) {
        remove_species(pattern, -delta);
    }
}

std::vector<std::string> NfsimSimulator::get_observable_names() const {
    impl_->require_session();
    return impl_->session_obs_names;
}

std::vector<double> NfsimSimulator::get_observable_values() const {
    impl_->require_session();
    auto *system = impl_->session_system;
    const int n_obs = impl_->session_n_obs;
    std::vector<double> values(n_obs);
    for (int i = 0; i < n_obs; ++i) {
        values[i] = static_cast<double>(system->getObsForOutput(i)->getCount());
    }
    return values;
}

double NfsimSimulator::get_parameter(const std::string &name) const {
    impl_->require_session();
    return impl_->session_system->getParameter(name);
}

bool NfsimSimulator::has_session() const { return impl_->session_system != nullptr; }

void NfsimSimulator::destroy_session() {
    if (impl_->session_system) {
        delete impl_->session_system;
        impl_->session_system = nullptr;
    }
    impl_->session_logical_time = 0.0;
    impl_->session_n_obs = 0;
    impl_->session_obs_names.clear();
    impl_->session_snapshots.clear();
}

void NfsimSimulator::save_species(const std::string &path) {
    impl_->require_session();
    StreamSuppressor suppress;
    // saveSpecies writes the BNG-format .species file: one line per
    // species pattern with its current count. The vendored impl prints
    // a progress line to cout before writing, suppressed here.
    impl_->session_system->saveSpecies(path);
}

void NfsimSimulator::save_concentrations(const std::string &label) {
    impl_->require_session();
    StreamSuppressor suppress;
    // Capture directly into our own per-label SystemSnapshot rather than
    // NFcore::System's single internal savedSnapshot, so distinct labels don't
    // clobber one another. Saving to an existing label replaces just that slot.
    auto snapshot = std::make_unique<::NFcore::SystemSnapshot>();
    snapshot->capture(impl_->session_system);
    impl_->session_snapshots[label] = std::move(snapshot);
}

void NfsimSimulator::restore_concentrations(const std::string &label) {
    impl_->require_session();
    auto it = impl_->session_snapshots.find(label);
    if (it == impl_->session_snapshots.end()) {
        throw std::runtime_error("NfsimSimulator: no saved concentrations to restore under label '" +
                                 label + "'. Call save_concentrations() first.");
    }
    StreamSuppressor suppress;
    // Reaction propensities and the live agent population change wholesale;
    // drop NFsim's stepTo cache before the restore so the next
    // simulate()/step_to() re-samples (NFcore::System::resetConcentrations does
    // this too; SystemSnapshot::restore() alone does not).
    impl_->session_system->invalidateStepToCache();
    it->second->restore(impl_->session_system);
}

bool NfsimSimulator::has_saved_concentrations(const std::string &label) const {
    return impl_->session_snapshots.find(label) != impl_->session_snapshots.end();
}

std::vector<std::string> NfsimSimulator::saved_concentration_labels() const {
    std::vector<std::string> labels;
    labels.reserve(impl_->session_snapshots.size());
    for (const auto &kv : impl_->session_snapshots) {
        labels.push_back(kv.first);
    }
    std::sort(labels.begin(), labels.end());
    return labels;
}

} // namespace bngsim
