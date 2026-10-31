// Absolute TFUN file paths for an NFsim load (issue #1314).
//
// NFsim opens a TFUN's `file=` at load (GlobalFunction::loadParamFile) by the
// path as written, so a relative one resolves against the process's working
// directory. bngsim used to change into the XML's directory around the load,
// which is process-wide: every other thread saw that directory for the
// duration, and an exception out of NFinput::initializeFromXML skipped the
// change back, leaving the host there. Instead, the XML NFsim reads gets each
// relative `file=` rewritten against the original XML's directory, so nothing
// about the process changes.

#pragma once

#include "seed_count_rounding.hpp" // detail::extract_attr, trim, make_temp_xml_path

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

namespace bngsim::tfunpath {

namespace detail {

// `s` as XML attribute text: what TinyXML decodes back to `s`.
inline std::string xml_escape(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
        case '&':
            out += "&amp;";
            break;
        case '<':
            out += "&lt;";
            break;
        case '>':
            out += "&gt;";
            break;
        case '"':
            out += "&quot;";
            break;
        case '\'':
            out += "&apos;";
            break;
        default:
            out += c;
        }
    }
    return out;
}

} // namespace detail

// Write a temp copy of `src_xml_path` in which every relative `file=` inside
// `<ListOfFunctions>` (a TFUN's table) is absolute, resolved against
// `base_dir`: the directory of the XML the user gave, which is not where an
// override or seed-round copy lives. nullopt when there is nothing to rewrite
// (no relative `file=`), or the copy cannot be written, in which case NFsim
// reads the path as written, as it always has.
//
// Only the base directory is inserted, escaped; the written value is kept as
// the XML holds it (its entities still encoded), so TinyXML decodes the pair
// to the path the user meant.
inline std::optional<std::filesystem::path>
write_tfun_absolute_xml(const std::string &src_xml_path, const std::filesystem::path &base_dir) {
    namespace fs = std::filesystem;
    namespace sd = bngsim::seedround::detail;
    try {
        std::ifstream in(src_xml_path);
        if (!in) {
            return std::nullopt;
        }
        const std::string prefix = detail::xml_escape((fs::absolute(base_dir) / "").string());
        std::vector<std::string> out_lines;
        std::string line;
        bool in_functions = false;
        bool changed = false;
        while (std::getline(in, line)) {
            if (!in_functions && line.find("<ListOfFunctions>") != std::string::npos) {
                in_functions = true;
            } else if (in_functions && line.find("</ListOfFunctions>") != std::string::npos) {
                in_functions = false;
            } else if (in_functions) {
                std::size_t vb = 0, ve = 0;
                if (auto value = sd::extract_attr(line, "file", &vb, &ve)) {
                    const std::string written = sd::trim(*value);
                    if (!written.empty() && fs::path(written).is_relative()) {
                        line = line.substr(0, vb) + prefix + written + line.substr(ve);
                        changed = true;
                    }
                }
            }
            out_lines.push_back(line);
        }
        if (!changed) {
            return std::nullopt;
        }
        fs::path tmp = sd::make_temp_xml_path();
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

} // namespace bngsim::tfunpath
