// bngsim/src/net_file_loader.cpp — .net file parser
//
// Parses BioNetGen .net files and feeds data into ModelBuilder.
// Format: begin parameters/species/reactions/groups/functions ... end ...
//
// Session 30: Refactored to use ModelBuilder as the sole construction path.
// NetFileLoader is now a thin parser — all model construction, optimization
// (ExprTk evaluator, stoichiometry, Jacobian sparsity, analytical Jacobian,
// graph coloring, SSA precompute) is handled by ModelBuilder::build().

#include "bngsim/net_file_loader.hpp"
#include "bngsim/model_builder.hpp"
#include "bngsim/types.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace bngsim {

// ─── Utility functions ───────────────────────────────────────────────────────

static std::string trim(const std::string &s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos)
        return "";
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

static std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> tokens;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, delim)) {
        token = trim(token);
        if (!token.empty())
            tokens.push_back(token);
    }
    return tokens;
}

static std::vector<std::string> split_ws(const std::string &s) {
    std::vector<std::string> tokens;
    std::istringstream iss(s);
    std::string token;
    while (iss >> token)
        tokens.push_back(token);
    return tokens;
}

// Strip trailing #comment. Returns trimmed content, optionally extracts comment.
static std::string strip_comment(const std::string &line, std::string *comment = nullptr) {
    int depth = 0;
    for (size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '(')
            ++depth;
        else if (line[i] == ')')
            --depth;
        else if (line[i] == '#' && depth == 0) {
            if (comment)
                *comment = trim(line.substr(i + 1));
            return trim(line.substr(0, i));
        }
    }
    if (comment)
        *comment = "";
    return trim(line);
}

static void skip_block(std::ifstream &file, const std::string &end_marker) {
    std::string line;
    while (std::getline(file, line)) {
        if (line.find(end_marker) != std::string::npos)
            break;
    }
}

static std::string directory_of(const std::string &filepath) {
    auto pos = filepath.find_last_of("/\\");
    if (pos == std::string::npos)
        return ".";
    return filepath.substr(0, pos);
}

// ─── Intermediate data structures for parsed blocks ──────────────────────────

struct ParsedParam {
    std::string name;
    double value;
    std::string expression;
    bool is_expression;
};

struct ParsedSpecies {
    std::string name;
    double concentration;
    bool fixed;
    bool is_param_ref;          // IC references a parameter name
    std::string param_ref_name; // parameter name for IC
};

struct ParsedFunction {
    std::string name;
    std::string expression;
};

struct ParsedObservable {
    std::string name;
    // Entries: (1-based species index, factor)
    std::vector<std::pair<int, double>> entries;
};

struct ParsedReaction {
    std::string comment;
    double stat_factor;
    std::vector<int> reactant_indices_1based; // 1-based species indices
    std::vector<int> product_indices_1based;  // 1-based species indices
    RateLawType type;
    std::string rate_law_name; // param or function name
    std::string legacy_rate_law_type;
    std::vector<std::string> legacy_rate_law_constants;
    // For MM: kcat_name, km_name
    std::string mm_kcat_name;
    std::string mm_km_name;
};

// ─── Parse blocks ────────────────────────────────────────────────────────────

static std::vector<ParsedParam> parse_parameters(std::ifstream &file) {
    std::vector<ParsedParam> params;
    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#')
            continue;
        if (line.find("end parameters") != std::string::npos)
            break;

        std::string stripped = strip_comment(line);
        auto tokens = split_ws(stripped);
        if (tokens.size() < 3)
            continue;

        ParsedParam p;
        p.name = tokens[1];

        // Value/expression: join remaining tokens (expressions may have spaces)
        std::string value_str = tokens[2];
        for (size_t i = 3; i < tokens.size(); ++i)
            value_str += tokens[i];

        p.expression = value_str;
        p.is_expression = false;

        try {
            size_t pos;
            p.value = std::stod(value_str, &pos);
            if (pos < value_str.size()) {
                p.is_expression = true;
            }
        } catch (...) {
            p.is_expression = true;
            p.value = 0.0;
        }

        params.push_back(std::move(p));
    }
    return params;
}

static std::vector<ParsedSpecies>
parse_species(std::ifstream &file, const std::unordered_map<std::string, int> &param_name_to_idx,
              const std::vector<ParsedParam> &params) {
    std::vector<ParsedSpecies> species;
    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#')
            continue;
        if (line.find("end species") != std::string::npos)
            break;

        std::string stripped = strip_comment(line);
        auto tokens = split_ws(stripped);
        if (tokens.size() < 3)
            continue;

        ParsedSpecies s;
        s.name = tokens[1];
        s.fixed = false;
        s.is_param_ref = false;

        // `$` clamp marker may sit at index 0 (`$Sink()`) or right after an
        // `@<compartment>::` prefix (`@CP::$Sink()`). BNG2.pl emits the
        // latter form for cBNGL models — issue #41.
        if (!s.name.empty() && s.name[0] == '$') {
            s.fixed = true;
            s.name = s.name.substr(1);
        } else if (!s.name.empty() && s.name[0] == '@') {
            auto sep = s.name.find("::");
            if (sep != std::string::npos && sep + 2 < s.name.size() && s.name[sep + 2] == '$') {
                s.fixed = true;
                s.name.erase(sep + 2, 1);
            }
        }

        // Concentration: number or parameter name
        std::string conc_str = tokens[2];
        try {
            size_t pos;
            s.concentration = std::stod(conc_str, &pos);
            if (pos < conc_str.size()) {
                s.is_param_ref = true;
                s.param_ref_name = conc_str;
            }
        } catch (...) {
            s.is_param_ref = true;
            s.param_ref_name = conc_str;
        }

        if (s.is_param_ref) {
            auto it = param_name_to_idx.find(conc_str);
            if (it != param_name_to_idx.end()) {
                s.concentration = params[it->second].value;
            } else {
                s.concentration = 0.0;
            }
        }

        species.push_back(std::move(s));
    }
    return species;
}

static std::vector<ParsedFunction> parse_functions(std::ifstream &file) {
    std::vector<ParsedFunction> functions;
    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#')
            continue;
        if (line.find("end functions") != std::string::npos)
            break;

        std::string stripped = strip_comment(line);
        auto tokens = split_ws(stripped);
        if (tokens.size() < 3)
            continue;

        ParsedFunction func;
        func.name = tokens[1];
        auto paren = func.name.find('(');
        if (paren != std::string::npos)
            func.name = func.name.substr(0, paren);

        func.expression = tokens[2];
        for (size_t i = 3; i < tokens.size(); ++i)
            func.expression += " " + tokens[i];

        functions.push_back(std::move(func));
    }
    return functions;
}

static std::vector<ParsedReaction> parse_reactions(std::ifstream &file) {
    std::vector<ParsedReaction> reactions;
    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#')
            continue;
        if (line.find("end reactions") != std::string::npos)
            break;

        std::string comment;
        std::string stripped = strip_comment(line, &comment);
        auto tokens = split_ws(stripped);
        if (tokens.size() < 4)
            continue;

        ParsedReaction rxn;
        rxn.comment = comment;
        rxn.stat_factor = 1.0;

        // Reactants (comma-separated species indices, 0 = null)
        for (const auto &rs : split(tokens[1], ',')) {
            int idx = std::stoi(rs);
            if (idx > 0)
                rxn.reactant_indices_1based.push_back(idx);
        }

        // Products
        for (const auto &ps : split(tokens[2], ',')) {
            int idx = std::stoi(ps);
            if (idx > 0)
                rxn.product_indices_1based.push_back(idx);
        }

        // Rate law
        std::string rate_token = tokens[3];

        if (rate_token == "Sat") {
            rxn.type = RateLawType::Functional;
            rxn.legacy_rate_law_type = rate_token;
            for (size_t i = 4; i < tokens.size(); ++i)
                rxn.legacy_rate_law_constants.push_back(tokens[i]);

        } else if (rate_token == "Hill") {
            rxn.type = RateLawType::Functional;
            rxn.legacy_rate_law_type = rate_token;
            for (size_t i = 4; i < tokens.size(); ++i)
                rxn.legacy_rate_law_constants.push_back(tokens[i]);

        } else if (rate_token == "MM" && tokens.size() >= 6) {
            rxn.type = RateLawType::MichaelisMenten;
            rxn.mm_kcat_name = tokens[4];
            rxn.mm_km_name = tokens[5];

        } else {
            // Elementary or Functional: check for "coeff*param" pattern
            std::string param_name = rate_token;
            auto star = rate_token.find('*');
            if (star != std::string::npos) {
                try {
                    rxn.stat_factor = std::stod(rate_token.substr(0, star));
                    param_name = rate_token.substr(star + 1);
                } catch (...) {
                    // Not coeff*name, treat as plain name
                }
            }

            rxn.type = RateLawType::Elementary;
            rxn.rate_law_name = param_name;
        }

        reactions.push_back(std::move(rxn));
    }
    return reactions;
}

static std::vector<ParsedObservable> parse_groups(std::ifstream &file) {
    std::vector<ParsedObservable> observables;
    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#')
            continue;
        if (line.find("end groups") != std::string::npos)
            break;

        std::string stripped = strip_comment(line);
        auto tokens = split_ws(stripped);
        if (tokens.size() < 2)
            continue; // need at least index and name

        ParsedObservable obs;
        obs.name = tokens[1];

        // Entries: comma-separated, possibly spread across tokens.
        if (tokens.size() >= 3) {
            std::string entries_str = tokens[2];
            for (size_t i = 3; i < tokens.size(); ++i)
                entries_str += tokens[i];

            for (const auto &es : split(entries_str, ',')) {
                double factor = 1.0;
                int sp_idx_1based = 0;
                auto star = es.find('*');
                if (star != std::string::npos) {
                    factor = std::stod(es.substr(0, star));
                    sp_idx_1based = std::stoi(es.substr(star + 1));
                } else {
                    sp_idx_1based = std::stoi(es);
                }
                obs.entries.emplace_back(sp_idx_1based, factor);
            }
        }

        observables.push_back(std::move(obs));
    }
    return observables;
}

// ─── tfun expression parsing helper ──────────────────────────────────────────
// Supports both file-based and inline tfun syntax used by BioNetGen.
//
// Supported patterns in .net functions block:
//   File-based:  tfun('file.tfun')
//                tfun('file.tfun', index)
//                tfun('file.tfun', index, method=>"step")
//   Inline:      tfun([x0,x1,...], [y0,y1,...], index)
//                tfun([x0,x1,...], [y0,y1,...], index, method=>"step")

struct TfunSpec {
    std::string filename; // non-empty for file-based
    std::string index_name;
    std::string method; // "linear" (default) or "step"
    // Inline data (populated when filename is empty)
    std::vector<double> xs;
    std::vector<double> ys;
    bool is_inline = false;
};

// Parse a bracket-delimited array: [1,2,3,4] → vector<double>
static std::vector<double> parse_bracket_array(const std::string &expr, size_t &pos) {
    std::vector<double> vals;
    if (pos >= expr.size() || expr[pos] != '[')
        return vals;
    ++pos; // skip '['
    while (pos < expr.size() && expr[pos] != ']') {
        // Skip whitespace
        while (pos < expr.size() && (expr[pos] == ' ' || expr[pos] == '\t'))
            ++pos;
        if (pos >= expr.size() || expr[pos] == ']')
            break;
        // Read number
        size_t num_start = pos;
        while (pos < expr.size() && expr[pos] != ',' && expr[pos] != ']' && expr[pos] != ' ')
            ++pos;
        std::string num_str = expr.substr(num_start, pos - num_start);
        if (!num_str.empty()) {
            try {
                vals.push_back(std::stod(num_str));
            } catch (...) {
                break;
            }
        }
        // Skip comma
        if (pos < expr.size() && expr[pos] == ',')
            ++pos;
    }
    if (pos < expr.size() && expr[pos] == ']')
        ++pos; // skip ']'
    return vals;
}

// Parse method=>"..." option from remaining expression
static std::string parse_method_option(const std::string &expr, size_t pos) {
    auto mpos = expr.find("method", pos);
    if (mpos == std::string::npos)
        return "linear";
    auto arrow = expr.find("=>", mpos);
    if (arrow == std::string::npos)
        return "linear";
    auto q1 = expr.find_first_of("'\"", arrow + 2);
    if (q1 == std::string::npos)
        return "linear";
    char quote = expr[q1];
    auto q2 = expr.find(quote, q1 + 1);
    if (q2 == std::string::npos)
        return "linear";
    return expr.substr(q1 + 1, q2 - q1 - 1);
}

static bool parse_tfun_expression(const std::string &expr, TfunSpec &spec) {
    auto pos = expr.find("tfun(");
    if (pos == std::string::npos)
        return false;

    size_t start = pos + 5; // after "tfun("

    // Skip whitespace
    while (start < expr.size() && (expr[start] == ' ' || expr[start] == '\t'))
        ++start;

    if (start >= expr.size())
        return false;

    // ── Inline mode: tfun([x,...], [y,...], index) ───────────────────────
    if (expr[start] == '[') {
        spec.is_inline = true;
        size_t p = start;
        spec.xs = parse_bracket_array(expr, p);
        // Skip comma + whitespace
        while (p < expr.size() && (expr[p] == ',' || expr[p] == ' ' || expr[p] == '\t'))
            ++p;
        spec.ys = parse_bracket_array(expr, p);
        // Skip comma + whitespace to get index name
        while (p < expr.size() && (expr[p] == ',' || expr[p] == ' ' || expr[p] == '\t'))
            ++p;
        // Read index name (until comma, paren, or whitespace)
        size_t idx_start = p;
        while (p < expr.size() && expr[p] != ',' && expr[p] != ')' && expr[p] != ' ' &&
               expr[p] != '\t')
            ++p;
        spec.index_name = (p > idx_start) ? expr.substr(idx_start, p - idx_start) : "time";
        spec.method = parse_method_option(expr, p);
        spec.filename = "";
        return !spec.xs.empty() && !spec.ys.empty();
    }

    // ── File-based mode: tfun('file.tfun', index) ────────────────────────
    auto q1 = expr.find_first_of("'\"", start);
    if (q1 == std::string::npos)
        return false;
    char quote = expr[q1];
    auto q2 = expr.find(quote, q1 + 1);
    if (q2 == std::string::npos)
        return false;

    spec.is_inline = false;
    spec.filename = expr.substr(q1 + 1, q2 - q1 - 1);
    spec.index_name = "time";
    spec.method = "linear";

    auto after_filename = expr.find_first_not_of(" \t", q2 + 1);
    if (after_filename != std::string::npos && expr[after_filename] == ',') {
        auto idx_start = expr.find_first_not_of(" \t", after_filename + 1);
        if (idx_start != std::string::npos) {
            // Check if next token is method=> or an index name
            if (expr.substr(idx_start, 6) == "method") {
                spec.method = parse_method_option(expr, idx_start);
            } else {
                auto idx_end = expr.find_first_of(" \t,)", idx_start);
                if (idx_end == std::string::npos)
                    idx_end = expr.size();
                spec.index_name = expr.substr(idx_start, idx_end - idx_start);
                // Check for method after index
                spec.method = parse_method_option(expr, idx_end);
            }
        }
    }

    return true;
}

// Located tfun(...) call inside a larger function expression.
struct TfunCallLocation {
    size_t start; // index of 't' in "tfun("
    size_t end;   // one past the matching ')'
    TfunSpec spec;
};

// Find the matching ')' for the '(' at expr[open_pos]. Returns std::string::npos
// if the parens are unbalanced.
static size_t find_matching_close_paren(const std::string &expr, size_t open_pos) {
    size_t depth = 1;
    for (size_t i = open_pos + 1; i < expr.size(); ++i) {
        if (expr[i] == '(')
            ++depth;
        else if (expr[i] == ')') {
            if (--depth == 0)
                return i;
        }
    }
    return std::string::npos;
}

// Find the next "tfun(" token starting at or after `pos`. The token must be a
// whole identifier (not part of a longer name like `mytfun(`).
static size_t find_next_tfun_token(const std::string &expr, size_t pos) {
    while (pos < expr.size()) {
        auto p = expr.find("tfun(", pos);
        if (p == std::string::npos)
            return std::string::npos;
        if (p == 0) {
            return p;
        }
        char prev = expr[p - 1];
        if (!(std::isalnum(static_cast<unsigned char>(prev)) || prev == '_'))
            return p;
        pos = p + 1;
    }
    return std::string::npos;
}

// Locate and parse every tfun(...) call inside `expr`. Caller decides whether
// to treat the result as a whole-body tfun (legacy path) or a set of embedded
// calls that require expression rewriting.
static std::vector<TfunCallLocation> find_all_tfun_calls(const std::string &expr) {
    std::vector<TfunCallLocation> out;
    size_t cursor = 0;
    while (cursor < expr.size()) {
        size_t start = find_next_tfun_token(expr, cursor);
        if (start == std::string::npos)
            break;
        size_t open_paren = start + 4; // points at '('
        size_t close_paren = find_matching_close_paren(expr, open_paren);
        if (close_paren == std::string::npos)
            break; // malformed — let downstream surface the error
        TfunCallLocation loc;
        loc.start = start;
        loc.end = close_paren + 1;
        std::string call_substr = expr.substr(start, loc.end - start);
        if (!parse_tfun_expression(call_substr, loc.spec))
            break; // can't parse — bail out so caller falls back gracefully
        out.push_back(std::move(loc));
        cursor = loc.end;
    }
    return out;
}

// True iff `expr`, modulo surrounding whitespace, is exactly a single tfun(...) call.
// Used to preserve the legacy "table function adopts the BNG function name" naming
// convention for backward compatibility (table_function_names asserts in tests).
static bool is_whole_body_tfun(const std::string &expr) {
    auto first = expr.find_first_not_of(" \t");
    auto last = expr.find_last_not_of(" \t");
    if (first == std::string::npos)
        return false;
    if (expr.compare(first, 5, "tfun(") != 0)
        return false;
    size_t close_paren = find_matching_close_paren(expr, first + 4);
    if (close_paren == std::string::npos)
        return false;
    return close_paren == last;
}

// ─── Legacy Sat/Hill .net canonicalization ──────────────────────────────────

static std::string unique_name(const std::string &base, std::unordered_set<std::string> &used) {
    std::string name = base;
    int suffix = 1;
    while (used.find(name) != used.end()) {
        name = base + "_" + std::to_string(suffix++);
    }
    used.insert(name);
    return name;
}

static std::vector<std::pair<int, int>>
reactant_multiplicities_preserving_order(const std::vector<int> &reactants_1based) {
    std::vector<std::pair<int, int>> out;
    for (int ri : reactants_1based) {
        auto it = std::find_if(out.begin(), out.end(),
                               [ri](const auto &entry) { return entry.first == ri; });
        if (it == out.end()) {
            out.emplace_back(ri, 1);
        } else {
            ++it->second;
        }
    }
    return out;
}

static std::string make_single_species_observable(int reaction_index, int species_index_1based,
                                                  int reactant_position_1based,
                                                  std::vector<ParsedObservable> &observables,
                                                  std::unordered_set<std::string> &used_obs_names) {
    std::string base = "__bngsim_net_rewrite_obs_r" + std::to_string(reaction_index) + "_" +
                       std::to_string(reactant_position_1based);
    std::string obs_name = unique_name(base, used_obs_names);

    ParsedObservable obs;
    obs.name = obs_name;
    obs.entries.emplace_back(species_index_1based, 1.0);
    observables.push_back(std::move(obs));
    return obs_name;
}

static std::string parenthesize_sum(const std::string &lhs, const std::string &rhs) {
    return "(" + lhs + " + " + rhs + ")";
}

static std::string parenthesize_product(const std::vector<std::string> &terms) {
    if (terms.empty())
        return "1";
    if (terms.size() == 1)
        return terms[0];

    std::string expr = "(";
    for (size_t i = 0; i < terms.size(); ++i) {
        if (i > 0)
            expr += " * ";
        expr += terms[i];
    }
    expr += ")";
    return expr;
}

static std::vector<std::string> rewrite_legacy_sat_hill_rate_laws(
    std::vector<ParsedReaction> &reactions, const std::vector<ParsedParam> &params,
    std::vector<ParsedFunction> &functions, std::vector<ParsedObservable> &observables) {
    std::unordered_set<std::string> used_function_names;
    for (const auto &func : functions)
        used_function_names.insert(func.name);
    for (const auto &param : params)
        used_function_names.insert(param.name);

    std::unordered_set<std::string> used_observable_names;
    for (const auto &obs : observables)
        used_observable_names.insert(obs.name);
    for (const auto &param : params)
        used_observable_names.insert(param.name);
    for (const auto &func : functions)
        used_observable_names.insert(func.name);

    int sat_count = 0;
    int hill_count = 0;

    for (int i = 0; i < static_cast<int>(reactions.size()); ++i) {
        auto &rxn = reactions[i];
        if (rxn.legacy_rate_law_type.empty())
            continue;

        const int reaction_index = i + 1;
        const auto unique_reactants =
            reactant_multiplicities_preserving_order(rxn.reactant_indices_1based);
        for (const auto &[species_index, count] : unique_reactants) {
            if (count != 1) {
                throw std::runtime_error(
                    "Legacy " + rxn.legacy_rate_law_type +
                    " .net rate law rewrite does not yet support repeated/non-unit "
                    "reactant stoichiometry in reaction " +
                    std::to_string(reaction_index) +
                    ". Rewrite this reaction as an explicit Functional rate law.");
            }
        }

        ParsedFunction func;
        func.name = unique_name("__bngsim_net_rewrite_" + rxn.legacy_rate_law_type + "_" +
                                    std::to_string(reaction_index),
                                used_function_names);

        if (rxn.legacy_rate_law_type == "Sat") {
            ++sat_count;
            if (rxn.legacy_rate_law_constants.size() < 2) {
                throw std::runtime_error(
                    "Saturation type .net rate laws require at least 2 rate constants "
                    "('Sat k K') in reaction " +
                    std::to_string(reaction_index));
            }
            const int n_saturation_constants =
                static_cast<int>(rxn.legacy_rate_law_constants.size()) - 1;
            if (unique_reactants.empty()) {
                throw std::runtime_error("Legacy Sat .net rate law requires at least one "
                                         "reactant in reaction " +
                                         std::to_string(reaction_index));
            }
            if (n_saturation_constants > static_cast<int>(unique_reactants.size())) {
                throw std::runtime_error(
                    "Saturation type .net rate laws cannot have more saturation constants "
                    "than reactants in reaction " +
                    std::to_string(reaction_index));
            }

            const std::string &k_name = rxn.legacy_rate_law_constants[0];
            std::vector<std::string> denominator_terms;
            for (int r = 0; r < n_saturation_constants; ++r) {
                std::string obs_name =
                    make_single_species_observable(reaction_index, unique_reactants[r].first, r + 1,
                                                   observables, used_observable_names);
                denominator_terms.push_back(
                    parenthesize_sum(rxn.legacy_rate_law_constants[r + 1], obs_name));
            }
            func.expression = k_name + " / " + parenthesize_product(denominator_terms);
        } else if (rxn.legacy_rate_law_type == "Hill") {
            ++hill_count;
            if (rxn.legacy_rate_law_constants.size() != 3) {
                throw std::runtime_error("Hill type .net rate laws require exactly 3 rate "
                                         "constants ('Hill Vmax Kh h') in reaction " +
                                         std::to_string(reaction_index));
            }
            if (unique_reactants.empty()) {
                throw std::runtime_error("Legacy Hill .net rate law requires at least one "
                                         "reactant in reaction " +
                                         std::to_string(reaction_index));
            }

            const std::string &vmax = rxn.legacy_rate_law_constants[0];
            const std::string &kh = rxn.legacy_rate_law_constants[1];
            const std::string &h = rxn.legacy_rate_law_constants[2];
            std::string obs_name = make_single_species_observable(
                reaction_index, unique_reactants[0].first, 1, observables, used_observable_names);

            func.expression = vmax + " * (" + obs_name + " ^ (" + h + " - 1)) / ((" + kh + " ^ " +
                              h + ") + (" + obs_name + " ^ " + h + "))";
        } else {
            throw std::runtime_error("Unsupported legacy .net rate law '" +
                                     rxn.legacy_rate_law_type + "'");
        }

        rxn.type = RateLawType::Functional;
        rxn.rate_law_name = func.name;
        functions.push_back(std::move(func));
    }

    std::vector<std::string> warnings;
    if (sat_count > 0 || hill_count > 0) {
        std::string warning = "Legacy/deprecated BioNetGen .net rate law token";
        if (sat_count + hill_count != 1)
            warning += "s";
        warning += " auto-rewritten by bngsim loader: ";
        bool need_sep = false;
        if (sat_count > 0) {
            warning += "Sat";
            if (sat_count > 1)
                warning += " (" + std::to_string(sat_count) + ")";
            need_sep = true;
        }
        if (hill_count > 0) {
            if (need_sep)
                warning += ", ";
            warning += "Hill";
            if (hill_count > 1)
                warning += " (" + std::to_string(hill_count) + ")";
        }
        warning += ". Rewrite the source BNGL as explicit Functional rate laws. "
                   "For a unit-stoichiometry reaction, use Sat k K as "
                   "f() = k/(K+S), not k*S/(K+S), because BNGL functional rates "
                   "multiply reactants separately. Use Hill Vmax Kh h as "
                   "f() = Vmax*S^(h-1)/(Kh^h+S^h).";
        warnings.push_back(std::move(warning));
    }
    return warnings;
}

// ─── Main loader ─────────────────────────────────────────────────────────────

NetworkModel NetFileLoader::load(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open .net file: " + path);
    }

    std::string net_file_dir = directory_of(path);

    // ── Phase 1: Parse all blocks into intermediate data ─────────────────
    std::vector<ParsedParam> parsed_params;
    std::vector<ParsedSpecies> parsed_species;
    std::vector<ParsedFunction> parsed_functions;
    std::vector<ParsedReaction> parsed_reactions;
    std::vector<ParsedObservable> parsed_observables;

    // Build param name→index map incrementally during parse for species IC resolution
    std::unordered_map<std::string, int> param_name_to_idx;

    std::string line;
    while (std::getline(file, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty() || trimmed[0] == '#')
            continue;
        if (trimmed.find("substanceUnits") != std::string::npos)
            continue;

        if (trimmed.find("begin parameters") != std::string::npos) {
            parsed_params = parse_parameters(file);
            for (int i = 0; i < static_cast<int>(parsed_params.size()); ++i) {
                param_name_to_idx[parsed_params[i].name] = i;
            }
        } else if (trimmed.find("begin species") != std::string::npos) {
            parsed_species = parse_species(file, param_name_to_idx, parsed_params);
        } else if (trimmed.find("begin functions") != std::string::npos) {
            parsed_functions = parse_functions(file);
        } else if (trimmed.find("begin reactions") != std::string::npos) {
            parsed_reactions = parse_reactions(file);
        } else if (trimmed.find("begin groups") != std::string::npos) {
            parsed_observables = parse_groups(file);
        } else if (trimmed.find("begin molecule types") != std::string::npos) {
            skip_block(file, "end molecule types");
        } else if (trimmed.find("begin observables") != std::string::npos) {
            skip_block(file, "end observables");
        } else if (trimmed.find("begin reaction rules") != std::string::npos) {
            skip_block(file, "end reaction rules");
        }
    }

    std::vector<std::string> load_warnings = rewrite_legacy_sat_hill_rate_laws(
        parsed_reactions, parsed_params, parsed_functions, parsed_observables);

    // ── Phase 2: Feed parsed data into ModelBuilder ──────────────────────
    ModelBuilder builder;
    builder.set_net_file_dir(net_file_dir);

    // 2a. Parameters
    for (const auto &p : parsed_params) {
        builder.add_parameter(p.name, p.value, p.expression, p.is_expression);
    }

    // 2b. Species
    for (int i = 0; i < static_cast<int>(parsed_species.size()); ++i) {
        const auto &s = parsed_species[i];
        builder.add_species(s.name, s.concentration, s.fixed);
        if (s.is_param_ref) {
            builder.add_species_param_ref(i, s.param_ref_name);
        }
    }

    // 2c. Functions — detect tfuns and register them.
    //
    // Two paths:
    // (a) Whole-body tfun (legacy): function expression IS a tfun(...) call.
    //     The table function adopts the BNG function name; ModelBuilder
    //     rewrites the function body to "tfun_<name>()" during build().
    // (b) Embedded tfun (wrapper-form or multi-tfun): function expression
    //     CONTAINS one or more tfun(...) calls inside arithmetic. Each call
    //     is registered as a synthetic anonymous table function named
    //     "<func>__tfun<k>"; the function expression is rewritten in-place
    //     so ExprTk sees "tfun_<func>__tfun<k>()" with wrapping arithmetic
    //     preserved.
    for (const auto &func : parsed_functions) {
        if (is_whole_body_tfun(func.expression)) {
            TfunSpec tfun_spec;
            if (parse_tfun_expression(func.expression, tfun_spec)) {
                builder.add_function(func.name, func.expression);
                if (tfun_spec.is_inline) {
                    builder.add_inline_table_function_spec(func.name, tfun_spec.xs, tfun_spec.ys,
                                                           tfun_spec.index_name, tfun_spec.method);
                } else {
                    builder.add_table_function_spec(func.name, tfun_spec.filename,
                                                    tfun_spec.index_name, tfun_spec.method);
                }
                continue;
            }
        }

        auto calls = find_all_tfun_calls(func.expression);
        if (calls.empty()) {
            builder.add_function(func.name, func.expression);
            continue;
        }

        std::string rewritten;
        rewritten.reserve(func.expression.size());
        size_t cursor = 0;
        for (size_t k = 0; k < calls.size(); ++k) {
            const auto &call = calls[k];
            rewritten.append(func.expression, cursor, call.start - cursor);
            std::string synth_name = func.name + "__tfun" + std::to_string(k);
            rewritten += "tfun_";
            rewritten += synth_name;
            rewritten += "()";
            cursor = call.end;
            if (call.spec.is_inline) {
                builder.add_inline_table_function_spec(synth_name, call.spec.xs, call.spec.ys,
                                                       call.spec.index_name, call.spec.method);
            } else {
                builder.add_table_function_spec(synth_name, call.spec.filename,
                                                call.spec.index_name, call.spec.method,
                                                /*header_name=*/func.name);
            }
        }
        rewritten.append(func.expression, cursor, std::string::npos);
        builder.add_function(func.name, rewritten);
    }

    // 2d. Observables (groups)
    for (const auto &obs : parsed_observables) {
        // Convert 1-based → 0-based species indices
        std::vector<std::pair<int, double>> entries_0based;
        entries_0based.reserve(obs.entries.size());
        for (const auto &[sp_idx_1, factor] : obs.entries) {
            entries_0based.emplace_back(sp_idx_1 - 1, factor);
        }
        builder.add_observable(obs.name, entries_0based);
    }

    // 2e. Reactions
    for (const auto &rxn : parsed_reactions) {
        // Convert 1-based → 0-based species indices
        std::vector<int> reactants_0based;
        for (int ri : rxn.reactant_indices_1based)
            reactants_0based.push_back(ri - 1);
        std::vector<int> products_0based;
        for (int pi : rxn.product_indices_1based)
            products_0based.push_back(pi - 1);

        if (rxn.type == RateLawType::MichaelisMenten) {
            // MM rate law: pass "kcat_name,km_name" as rate_law string
            std::string mm_rate = rxn.mm_kcat_name + "," + rxn.mm_km_name;
            builder.add_reaction(reactants_0based, products_0based, RateLawType::MichaelisMenten,
                                 mm_rate, rxn.stat_factor);
        } else {
            // Elementary or Functional (resolved during build())
            builder.add_reaction(reactants_0based, products_0based, rxn.type, rxn.rate_law_name,
                                 rxn.stat_factor);
        }
    }

    // ── Phase 3: Build the model ─────────────────────────────────────────
    // ModelBuilder::build() handles ALL post-parse processing:
    // - ExprTk evaluator setup (parameter/observable/function registration)
    // - Synthetic parameter creation for functions
    // - Expression compilation (parameters, functions)
    // - Inter-function reference resolution (funcName() → funcName)
    // - Species IC re-resolution from parameter values
    // - Table function loading and registration
    // - Functional reaction type resolution
    // - Stoichiometry matrix construction
    // - Jacobian sparsity pattern computation
    // - Analytical Jacobian pre-computation
    // - Graph coloring (Curtis-Powell-Reid)
    // - SSA propensity pre-computation
    NetworkModel model = builder.build();
    model.set_load_warnings_(std::move(load_warnings));
    return model;
}

} // namespace bngsim
