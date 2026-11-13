#pragma once

// Issue #1653 — text read from a file, on its way to Python.
//
// pybind11 hands every name, expression and message the core makes to Python
// through Python's strict UTF-8 decoder, which a file's bytes never met: a
// Latin-1 `.net` species `Aµ()` (the byte 0xb5) loaded, and its name raised a
// bare UnicodeDecodeError that named neither the file nor the line. A refusal
// quoting such a byte, from the `.net` loader or the table-function reader,
// lost its words to the same error.

#include <cstddef>
#include <string>

namespace bngsim {

// The length of the UTF-8 character that starts at s[i], or 0 where none does:
// the well-formed byte sequences of the Unicode Standard's Table 3-7, which
// are the ones Python's strict decoder takes (no overlong form, no surrogate,
// nothing past U+10FFFF).
inline std::size_t utf8_char_length(const std::string &s, std::size_t i) {
    const auto at = [&s](std::size_t k) { return static_cast<unsigned char>(s[k]); };
    const unsigned char c = at(i);
    if (c < 0x80)
        return 1;
    std::size_t n = 0;
    unsigned char lo = 0x80, hi = 0xBF; // the range of the second byte
    if (c >= 0xC2 && c <= 0xDF) {
        n = 2;
    } else if (c == 0xE0) {
        n = 3;
        lo = 0xA0;
    } else if ((c >= 0xE1 && c <= 0xEC) || c == 0xEE || c == 0xEF) {
        n = 3;
    } else if (c == 0xED) {
        n = 3;
        hi = 0x9F;
    } else if (c == 0xF0) {
        n = 4;
        lo = 0x90;
    } else if (c >= 0xF1 && c <= 0xF3) {
        n = 4;
    } else if (c == 0xF4) {
        n = 4;
        hi = 0x8F;
    } else {
        return 0;
    }
    if (i + n > s.size() || at(i + 1) < lo || at(i + 1) > hi)
        return 0;
    for (std::size_t k = 2; k < n; ++k) {
        if ((at(i + k) & 0xC0) != 0x80)
            return 0;
    }
    return n;
}

// The byte `b` as two lowercase hex digits, `b5`.
inline std::string hex_byte(unsigned char b) {
    static const char kHex[] = "0123456789abcdef";
    return std::string{kHex[b >> 4], kHex[b & 0xF]};
}

// `s`, with each byte that is not part of a UTF-8 character shown as `\xhh`,
// as Python's "backslashreplace" shows it. UTF-8 text comes back unchanged.
inline std::string shown_as_utf8(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (std::size_t i = 0; i < s.size();) {
        const std::size_t n = utf8_char_length(s, i);
        if (n == 0) {
            out += "\\x" + hex_byte(static_cast<unsigned char>(s[i]));
            ++i;
        } else {
            out.append(s, i, n);
            i += n;
        }
    }
    return out;
}

} // namespace bngsim
