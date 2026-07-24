#ifndef BNGSIM_PARAM_OVERRIDE_XML_HPP
#define BNGSIM_PARAM_OVERRIDE_XML_HPP

// Shared BNG XML <Parameter> table + override propagation for the network-free
// session backends (NFsim, RuleMonkey).
//
// BNG2.pl emits every <Parameter> as both a precomputed `value=` and its
// symbolic `expr=`. The vendored engines record only `value=` and drop `expr=`,
// so a session set_param() on a base parameter leaves derived parameters
// (`Ntot = 100*scale`, `LT = LT_conc_M*NA*V_sim`, `_rateLawN`, ...) pinned at
// their XML-time values. That is silently wrong for a parameter scan or a
// multi-phase network-free protocol.
//
// This header closes the gap without patching either vendored loader: it parses
// every <Parameter> once, keeps the BNG-emitted expression strings, and
// re-evaluates the whole table through bngsim's ExprTk evaluator whenever
// overrides change. `write_param_overridden_xml` then bakes the propagated
// values back into a temp copy of the XML's `<Parameter value=>` attributes, so
// the engine picks up the overridden namespace when it parses seed-species
// concentrations and rate constants.
//
// The NFsim backend (GH #20 / #29) and the RuleMonkey backend (GH #44) both use
// this. Kept in its own translation-unit-free header so both wrappers share one
// implementation.

#include "bngsim/expression.hpp"
#include "bngsim/platform_compat.hpp" // POSIX getpid() shim for Windows (GH #150)

#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace bngsim::paramxml {

inline std::string trim_ascii(std::string s) {
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

// Extract a quoted attribute value: name="..." (or name='...'). BNG2.pl uses
// double quotes throughout; supporting both keeps this resilient to any future
// XML-emitter tweaks. Matches `attr` as a whole token, not as the suffix of some
// other attribute. Returns the inner string and (out-params) the
// [open_quote_index, close_quote_index] span so callers can splice replacement
// text into the original line without re-scanning.
inline std::optional<std::string> extract_xml_attr(const std::string &line, const std::string &attr,
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

// Re-evaluate every parameter through ExprTk with `overrides` applied.
//
// Returns name→value for every parameter in the table (overrides included).
// Evaluation walks the table in declaration order — BNG2.pl emits parameters
// such that every reference is to a previously-declared name, so a single
// pass produces a fixed point.
//
// Throws std::runtime_error if any parameter expression fails to compile or
// evaluate; that is intentionally loud because silently leaving downstream
// parameters at stale values is exactly the bug this closes.
inline std::unordered_map<std::string, double>
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

// Write a per-process unique temp XML alongside the system temp dir. Caller owns
// the returned path and must remove the file when done.
//
// Why a temp file (and not in-memory string): the vendored loaders take a path.
// The cost (one ~50 KB write + one ~50 KB read) is negligible next to parsing.
inline std::filesystem::path make_unique_tmp_xml_path() {
    namespace fs = std::filesystem;
    static std::atomic<uint64_t> counter{0};
    const auto pid = static_cast<long>(::getpid());
    const auto n = counter.fetch_add(1, std::memory_order_relaxed);
    fs::path tmp_dir = fs::temp_directory_path();
    return tmp_dir / ("bngsim_paramxml_" + std::to_string(pid) + "_" + std::to_string(n) + ".xml");
}

// Format an override-resolved value with full IEEE 754 precision so the
// rewritten XML round-trips exactly to the same bit pattern. The vendored
// loaders parse `value=` via strtod, which accepts `%.17g` output.
inline std::string format_override_value(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return std::string(buf);
}

// Splice a new `value="..."` attribute payload into a `<Parameter ...>` line,
// leaving everything else (including any `expr="..."` attribute) untouched.
// Returns the line unchanged if `value=` is missing or malformed.
inline std::string rewrite_parameter_value_attr(const std::string &line, double new_value) {
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
// pointers) is preserved verbatim — the loaders resolve those through the
// parameter map, so rewriting `<Parameter value=>` cascades automatically.
//
// Throws std::runtime_error on I/O failure. Caller must remove the
// returned path when done.
inline std::filesystem::path
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

} // namespace bngsim::paramxml

#endif // BNGSIM_PARAM_OVERRIDE_XML_HPP
