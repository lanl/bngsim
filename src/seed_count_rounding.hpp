#ifndef BNGSIM_SEED_COUNT_ROUNDING_HPP
#define BNGSIM_SEED_COUNT_ROUNDING_HPP

// Single source of bngsim's cold-start stochastic seed-count rounding policy
// (GH #51).
//
// The network-free engines (NFsim, RuleMonkey) parse `<Species
// concentration="X">` out of BNG XML and integerize X by truncation toward zero
// (a `(int)` cast in their vendored loaders), so 5.7 seeds 5 molecules. bngsim
// SSA, BNG2.pl `run_network` (Network3), and COPASI's stochastic solvers all
// round *half-up* instead — nearest integer, ties away from zero — so 5.7 seeds
// 6. To make the network-free seed agree with the rest of the stack *without*
// patching vendored loaders (so the fix survives a vendor refresh), bngsim
// rounds each fractional seed concentration here, in the XML it hands to the
// engine: the rounded value is already an integer, so the vendored `(int)` cast
// becomes a no-op.
//
// This is the C++ cold-start arm of one repo-wide policy. The Python form is
// `bngsim._rounding.round_half_up` (session count mutations + the PyBNF NF
// bridge); the SSA form is `round_initial_population_to_storage` in
// `src/ssa_simulator.cpp`. Keep all three in sync.
//
// Everything here is best-effort and fail-safe: any token that cannot be
// resolved to a finite value (a compound expression, an unknown id) and any I/O
// error leaves the source untouched, so a model that loads today still loads —
// at worst it keeps the legacy truncating behavior.

#include "bngsim/platform_compat.hpp" // POSIX getpid() shim for Windows (GH #150)

#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace bngsim::seedround {

// Round-half-up (nearest integer, ties away from zero) — bngsim's repo-wide
// stochastic initial-count policy (GH #51). Idempotent on integers.
inline long long round_half_up_count(double x) {
    return (x >= 0.0) ? static_cast<long long>(std::floor(x + 0.5))
                      : static_cast<long long>(std::ceil(x - 0.5));
}

namespace detail {

inline std::string trim(const std::string &s) {
    std::size_t b = 0, e = s.size();
    while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) {
        ++b;
    }
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {
        --e;
    }
    return s.substr(b, e - b);
}

// Value of attribute `attr` in `line` (text between the quotes), plus the
// [*vbegin, *vend) span of that text when requested. Matches `attr` as a whole
// token, not as a suffix of some other attribute. Mirrors `extract_xml_attr` in
// nfsim_simulator.cpp but kept self-contained here so this header has no
// translation-unit dependencies.
inline std::optional<std::string> extract_attr(const std::string &line, const std::string &attr,
                                               std::size_t *vbegin = nullptr,
                                               std::size_t *vend = nullptr) {
    const std::string needle = attr + "=";
    std::size_t pos = 0;
    while ((pos = line.find(needle, pos)) != std::string::npos) {
        if (pos > 0) {
            char prev = line[pos - 1];
            if (std::isalnum(static_cast<unsigned char>(prev)) || prev == '_') {
                pos += needle.size();
                continue;
            }
        }
        std::size_t q = pos + needle.size();
        if (q >= line.size() || (line[q] != '"' && line[q] != '\'')) {
            return std::nullopt;
        }
        char quote = line[q];
        std::size_t end = line.find(quote, q + 1);
        if (end == std::string::npos) {
            return std::nullopt;
        }
        if (vbegin) {
            *vbegin = q + 1;
        }
        if (vend) {
            *vend = end;
        }
        return line.substr(q + 1, end - q - 1);
    }
    return std::nullopt;
}

// Resolve a concentration token to a finite double: first as a numeric literal
// (covers `5.7`, `2e4`, ...), then as a parameter id looked up in `overlay`
// (override-resolved values, wins) or `xml_values` (the XML's own `value=`
// attrs). Compound expressions and unknown ids return nullopt.
inline std::optional<double>
resolve_token(const std::string &tok, const std::unordered_map<std::string, double> &overlay,
              const std::unordered_map<std::string, double> &xml_values) {
    std::string t = trim(tok);
    if (t.empty()) {
        return std::nullopt;
    }
    try {
        std::size_t consumed = 0;
        double d = std::stod(t, &consumed);
        if (consumed == t.size() && std::isfinite(d)) {
            return d;
        }
    } catch (const std::exception &) {
        // Not a numeric literal; fall through to a parameter lookup.
    }
    if (auto it = overlay.find(t); it != overlay.end() && std::isfinite(it->second)) {
        return it->second;
    }
    if (auto it = xml_values.find(t); it != xml_values.end() && std::isfinite(it->second)) {
        return it->second;
    }
    return std::nullopt;
}

inline std::filesystem::path make_temp_xml_path() {
    namespace fs = std::filesystem;
    static std::atomic<std::uint64_t> counter{0};
    const auto pid = static_cast<long>(::getpid());
    const auto n = counter.fetch_add(1, std::memory_order_relaxed);
    return fs::temp_directory_path() /
           ("bngsim_seedround_" + std::to_string(pid) + "_" + std::to_string(n) + ".xml");
}

} // namespace detail

// RAII owner for a temp XML produced by write_seed_rounded_xml: best-effort
// removal of the file on destruction. Declare it *before* the consumer that
// reads the path (e.g. a simulator constructed from it) so the file outlives
// every read — class members are destroyed in reverse declaration order, so a
// guard declared first is destroyed last.
class TempXmlFile {
  public:
    explicit TempXmlFile(std::optional<std::filesystem::path> path = std::nullopt)
        : path_(std::move(path)) {}
    ~TempXmlFile() {
        if (path_) {
            std::error_code ec;
            std::filesystem::remove(*path_, ec);
        }
    }
    TempXmlFile(const TempXmlFile &) = delete;
    TempXmlFile &operator=(const TempXmlFile &) = delete;
    TempXmlFile(TempXmlFile &&) = default;
    TempXmlFile &operator=(TempXmlFile &&) = default;

    bool active() const { return path_.has_value(); }
    // The temp path's string if rounding produced one, else `fallback`.
    std::string path_or(const std::string &fallback) const {
        return path_ ? path_->string() : fallback;
    }

  private:
    std::optional<std::filesystem::path> path_;
};

// Write a temp copy of `src_xml_path` in which every `<Species ...
// concentration="TOK">` whose TOK resolves to a finite, *non-integer* value has
// its `concentration=` attribute rewritten to that value rounded half-up
// (GH #51). TOK is resolved as a numeric literal, or as a parameter id looked up
// in `param_value_overlay` (override-resolved values, e.g. NFsim's pre-init
// setParameter propagation) or the XML's own `<Parameter value="...">` attrs.
//
// Returns the temp path if any concentration was rewritten; std::nullopt if the
// copy would be byte-identical to the source (caller then loads the original and
// pays no temp-file cost) or on any error (fail-safe: legacy truncation stands).
// The caller owns the returned temp file and must remove it when done.
inline std::optional<std::filesystem::path>
write_seed_rounded_xml(const std::string &src_xml_path,
                       const std::unordered_map<std::string, double> &param_value_overlay = {}) {
    namespace fs = std::filesystem;
    try {
        std::ifstream in(src_xml_path);
        if (!in) {
            return std::nullopt;
        }

        // BNG XML lists parameters before species, so a single forward pass can
        // build the id→value map and then resolve concentrations against it.
        std::unordered_map<std::string, double> xml_values;
        std::vector<std::string> out_lines;
        std::string line;
        bool in_param_list = false;
        bool in_species_list = false;
        bool changed = false;

        while (std::getline(in, line)) {
            const std::string trimmed = detail::trim(line);

            if (!in_param_list && line.find("<ListOfParameters>") != std::string::npos) {
                in_param_list = true;
            } else if (in_param_list && line.find("</ListOfParameters>") != std::string::npos) {
                in_param_list = false;
            } else if (in_param_list && trimmed.rfind("<Parameter", 0) == 0) {
                if (auto id = detail::extract_attr(trimmed, "id")) {
                    if (auto v = detail::extract_attr(trimmed, "value")) {
                        try {
                            xml_values[*id] = std::stod(*v);
                        } catch (const std::exception &) {
                            // Non-numeric value= (shouldn't occur in BNG XML); skip.
                        }
                    }
                }
            }

            if (!in_species_list && line.find("<ListOfSpecies>") != std::string::npos) {
                in_species_list = true;
            } else if (in_species_list && line.find("</ListOfSpecies>") != std::string::npos) {
                in_species_list = false;
            } else if (in_species_list && trimmed.rfind("<Species", 0) == 0) {
                std::size_t vb = 0, ve = 0;
                if (auto tok = detail::extract_attr(line, "concentration", &vb, &ve)) {
                    if (auto value = detail::resolve_token(*tok, param_value_overlay, xml_values)) {
                        if (std::floor(*value) != *value) {
                            const long long rounded = round_half_up_count(*value);
                            line = line.substr(0, vb) + std::to_string(rounded) + line.substr(ve);
                            changed = true;
                        }
                    }
                }
            }

            out_lines.push_back(line);
        }

        if (!changed) {
            return std::nullopt;
        }

        fs::path tmp = detail::make_temp_xml_path();
        std::ofstream out(tmp);
        if (!out) {
            return std::nullopt;
        }
        for (const auto &l : out_lines) {
            out << l << '\n';
        }
        out.close();
        if (!out) {
            std::error_code ec;
            fs::remove(tmp, ec);
            return std::nullopt;
        }
        return tmp;
    } catch (const std::exception &) {
        return std::nullopt;
    }
}

} // namespace bngsim::seedround

#endif // BNGSIM_SEED_COUNT_ROUNDING_HPP
