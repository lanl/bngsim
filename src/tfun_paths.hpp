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

#include "seed_count_rounding.hpp" // detail::trim, make_temp_xml_path

#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
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

namespace detail {

inline bool is_xml_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

// A character of an XML name, as TinyXML's ReadName takes one.
inline bool is_xml_name_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.' ||
           c == ':';
}

// Whether the tag `<name` (or `</name` with `closing`) starts at `pos`, the name
// ending there rather than running on into a longer one.
inline bool tag_named_at(const std::string &text, std::size_t pos, const std::string &name,
                         bool closing = false) {
    const std::string open = closing ? "</" + name : "<" + name;
    if (text.compare(pos, open.size(), open) != 0) {
        return false;
    }
    const std::size_t after = pos + open.size();
    return after < text.size() && !is_xml_name_char(text[after]);
}

// An attribute of a start tag: its name, and the [begin, end) span of its value
// in the text, between the quotes, with `quote` 0 for an unquoted value.
struct Attribute {
    std::string name;
    std::size_t begin = 0;
    std::size_t end = 0;
    char quote = 0;
};

// The attributes of the start tag whose name ends at `pos`, read as TinyXML's
// TiXmlAttribute::Parse reads them: whitespace round the `=`, a value in single
// or double quotes, or unquoted up to whitespace, `/` or `>`. `*tag_end` is the
// index of the `>` that closes the tag (text.size() if none), `*self_closing`
// whether it is `/>`.
inline std::vector<Attribute> read_attributes(const std::string &text, std::size_t pos,
                                              std::size_t *tag_end, bool *self_closing) {
    std::vector<Attribute> attrs;
    std::size_t i = pos;
    *self_closing = false;
    while (i < text.size()) {
        while (i < text.size() && is_xml_space(text[i])) {
            ++i;
        }
        if (i >= text.size()) {
            break;
        }
        if (text[i] == '>') {
            *tag_end = i;
            return attrs;
        }
        if (text[i] == '/') {
            if (i + 1 < text.size() && text[i + 1] == '>') {
                *self_closing = true;
                *tag_end = i + 1;
                return attrs;
            }
            ++i;
            continue;
        }
        const std::size_t name_begin = i;
        while (i < text.size() && is_xml_name_char(text[i])) {
            ++i;
        }
        if (i == name_begin) {
            ++i; // not a name: step over it, as no attribute starts here
            continue;
        }
        Attribute a;
        a.name = text.substr(name_begin, i - name_begin);
        while (i < text.size() && is_xml_space(text[i])) {
            ++i;
        }
        if (i >= text.size() || text[i] != '=') {
            continue; // a name with no value: TinyXML refuses the document
        }
        ++i;
        while (i < text.size() && is_xml_space(text[i])) {
            ++i;
        }
        if (i < text.size() && (text[i] == '"' || text[i] == '\'')) {
            a.quote = text[i];
            a.begin = i + 1;
            const std::size_t close = text.find(a.quote, a.begin);
            a.end = close == std::string::npos ? text.size() : close;
            i = close == std::string::npos ? text.size() : close + 1;
        } else {
            a.begin = i;
            while (i < text.size() && !is_xml_space(text[i]) && text[i] != '/' && text[i] != '>') {
                ++i;
            }
            a.end = i;
        }
        attrs.push_back(std::move(a));
    }
    *tag_end = text.size();
    return attrs;
}

} // namespace detail

// Write a temp copy of `src_xml_path` in which every relative `file=` of a
// `<Function>` inside `<ListOfFunctions>` (a TFUN's table) is absolute, resolved
// against `base_dir`: the directory of the XML the user gave, which is not where
// an override or seed-round copy lives. nullopt when there is nothing to rewrite
// (no relative `file=`), or the copy cannot be written, in which case NFsim
// reads the path as written, as it always has.
//
// The XML is read as NFsim's TinyXML reads it, not by line: a tag may share a
// line with others or run over several, and `file = '…'`, `file="…"` and an
// unquoted `file=…` are all one attribute. A path the scan missed would be
// opened against the process's working directory, where a table of the same
// name is the wrong one.
//
// Only the base directory is inserted, escaped; the written value is kept as
// the XML holds it (its entities still encoded), so TinyXML decodes the pair
// to the path the user meant. A path with a root (`/data/t.dat`, and on Windows
// `C:t.dat` too) is left as written.
inline std::optional<std::filesystem::path>
write_tfun_absolute_xml(const std::string &src_xml_path, const std::filesystem::path &base_dir) {
    namespace fs = std::filesystem;
    namespace sd = bngsim::seedround::detail;
    try {
        std::ifstream in(src_xml_path, std::ios::binary);
        if (!in) {
            return std::nullopt;
        }
        const std::string text((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
        const std::string prefix = detail::xml_escape((fs::absolute(base_dir) / "").string());
        std::string out;
        out.reserve(text.size() + 256);
        std::size_t copied = 0;
        bool in_functions = false;
        bool changed = false;
        std::size_t i = 0;
        while (i < text.size()) {
            if (text[i] != '<') {
                ++i;
                continue;
            }
            if (text.compare(i, 4, "<!--") == 0) {
                const std::size_t e = text.find("-->", i + 4);
                i = e == std::string::npos ? text.size() : e + 3;
                continue;
            }
            if (text.compare(i, 9, "<![CDATA[") == 0) {
                const std::size_t e = text.find("]]>", i + 9);
                i = e == std::string::npos ? text.size() : e + 3;
                continue;
            }
            if (detail::tag_named_at(text, i, "ListOfFunctions", /*closing=*/true)) {
                in_functions = false;
                ++i;
                continue;
            }
            const bool list = detail::tag_named_at(text, i, "ListOfFunctions");
            const bool function = in_functions && detail::tag_named_at(text, i, "Function");
            if (!list && !function) {
                ++i;
                continue;
            }
            const std::size_t name_end = i + 1 + (list ? 15 : 8);
            std::size_t tag_end = 0;
            bool self_closing = false;
            const auto attrs = detail::read_attributes(text, name_end, &tag_end, &self_closing);
            if (list) {
                in_functions = !self_closing;
            } else {
                for (const auto &a : attrs) {
                    if (a.name != "file") {
                        continue;
                    }
                    const std::string written = sd::trim(text.substr(a.begin, a.end - a.begin));
                    if (written.empty() || fs::path(written).has_root_path()) {
                        continue;
                    }
                    out.append(text, copied, a.begin - copied);
                    if (a.quote) {
                        out += prefix + written;
                    } else {
                        out += '"' + prefix + written + '"';
                    }
                    copied = a.end;
                    changed = true;
                }
            }
            i = tag_end < text.size() ? tag_end + 1 : text.size();
        }
        if (!changed) {
            return std::nullopt;
        }
        out.append(text, copied, std::string::npos);
        fs::path tmp = sd::make_temp_xml_path();
        std::ofstream outf(tmp, std::ios::binary);
        if (!outf) {
            return std::nullopt;
        }
        outf << out;
        outf.close();
        if (!outf) {
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
