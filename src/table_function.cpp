// bngsim/src/table_function.cpp — Table function (tfun) implementation
//
// Piecewise-linear or step interpolation of tabular data.
// Binary search + interpolation with constant extrapolation beyond endpoints.
// Supports both file-based tfuns and inline tfun array syntax.

#include "bngsim/table_function.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace bngsim {

// ─── Constructor ─────────────────────────────────────────────────────────────

TableFunction::TableFunction(const std::string &name, std::vector<double> xs,
                             std::vector<double> ys, const std::string &index_name,
                             InterpolationMethod method)
    : name_(name), index_name_(index_name), xs_(std::move(xs)), ys_(std::move(ys)),
      method_(method) {

    if (xs_.size() != ys_.size()) {
        throw std::runtime_error(
            "TableFunction '" + name_ + "': x and y vectors must have the same size (" +
            std::to_string(xs_.size()) + " vs " + std::to_string(ys_.size()) + ")");
    }
    if (xs_.size() < 2) {
        throw std::runtime_error("TableFunction '" + name_ +
                                 "': need at least 2 data points, got " +
                                 std::to_string(xs_.size()));
    }

    // Check monotonically increasing x values
    for (size_t i = 1; i < xs_.size(); ++i) {
        if (xs_[i] <= xs_[i - 1]) {
            throw std::runtime_error("TableFunction '" + name_ +
                                     "': x values must be monotonically "
                                     "increasing. Found x[" +
                                     std::to_string(i - 1) + "]=" + std::to_string(xs_[i - 1]) +
                                     " >= x[" + std::to_string(i) + "]=" + std::to_string(xs_[i]));
        }
    }
}

// ─── Evaluate ────────────────────────────────────────────────────────────────

double TableFunction::evaluate_at(double x) const {
    // Constant extrapolation beyond endpoints
    if (x <= xs_.front())
        return ys_.front();
    if (x >= xs_.back())
        return ys_.back();

    // Binary search for the interval containing x
    // Find the first element > x, then back up one
    auto it = std::upper_bound(xs_.begin(), xs_.end(), x);
    int i = static_cast<int>(it - xs_.begin()) - 1;

    // i is the index of the left endpoint of the interval [xs_[i], xs_[i+1])
    if (method_ == InterpolationMethod::Step) {
        // Step function (left-continuous): return value at left endpoint
        return ys_[i];
    }

    // Piecewise-linear interpolation (default)
    double x0 = xs_[i];
    double x1 = xs_[i + 1];
    double y0 = ys_[i];
    double y1 = ys_[i + 1];

    double frac = (x - x0) / (x1 - x0);
    return y0 + frac * (y1 - y0);
}

double TableFunction::evaluate() const {
    if (index_ptr_ == nullptr) {
        return ys_.front(); // fallback: return first value
    }
    return evaluate_at(*index_ptr_);
}

// ─── File loader ─────────────────────────────────────────────────────────────

// Shared canonicalization helpers (`is_time_index`, `strip_paren_suffix`) are
// defined in include/bngsim/table_function.hpp so the .tfun reader and the
// runtime index resolver agree on what counts as a time index and how trailing
// "()" suffixes are normalized.

TableFunction TableFunction::from_file(const std::string &name, const std::string &filepath,
                                       const std::string &index_name, InterpolationMethod method,
                                       const std::string &header_name) {
    const std::string &expected_header = header_name.empty() ? name : header_name;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("TableFunction '" + name + "': cannot open file '" + filepath +
                                 "'");
    }

    std::vector<double> xs, ys;
    bool header_found = false;
    std::string line;

    while (std::getline(file, line)) {
        // Trim leading/trailing whitespace
        auto start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos)
            continue; // empty line
        auto end = line.find_last_not_of(" \t\r\n");
        line = line.substr(start, end - start + 1);

        if (line.empty())
            continue;

        if (line[0] == '#') {
            if (!header_found) {
                // Parse header: "# col1_name  col2_name"
                std::string header = line.substr(1); // strip '#'
                std::istringstream hss(header);
                std::string col1, col2;
                hss >> col1 >> col2;

                if (col1.empty() || col2.empty()) {
                    throw std::runtime_error("TableFunction '" + name +
                                             "': header line must have "
                                             "two column names (got: '" +
                                             line + "')");
                }

                // Validate column 1: must match index name.
                // For the time index, accept case-insensitive time/t with or
                // without trailing "()". For param/obs indices, strip a
                // trailing "()" from the header before comparing (matches
                // BNG's TfunReader.pm:84-93 normalization).
                bool col1_ok = false;
                if (is_time_index(index_name)) {
                    col1_ok = is_time_index(col1);
                } else {
                    col1_ok = (strip_paren_suffix(col1) == strip_paren_suffix(index_name));
                }
                if (!col1_ok) {
                    throw std::runtime_error("TableFunction '" + name + "': header column 1 '" +
                                             col1 + "' does not match expected index name '" +
                                             index_name + "'");
                }

                // Validate column 2: must match the expected header label
                // (after stripping a trailing "()"; BNG TfunReader.pm:97 does
                // the same on both columns). The header_name parameter lets
                // wrapper-form tfun (synthetic runtime ID) validate against
                // the original BNG function name.
                if (strip_paren_suffix(col2) != strip_paren_suffix(expected_header)) {
                    throw std::runtime_error("TableFunction '" + name + "': header column 2 '" +
                                             col2 + "' does not match expected function name '" +
                                             expected_header + "'");
                }

                header_found = true;
            }
            continue; // skip all comment lines
        }

        // Data line: two whitespace-separated numbers
        std::istringstream dss(line);
        double x, y;
        if (!(dss >> x >> y)) {
            throw std::runtime_error("TableFunction '" + name + "': invalid data line: '" + line +
                                     "'");
        }

        xs.push_back(x);
        ys.push_back(y);
    }

    if (!header_found) {
        throw std::runtime_error("TableFunction '" + name + "': file '" + filepath +
                                 "' has no '#'-prefixed header line");
    }

    if (xs.size() < 2) {
        throw std::runtime_error("TableFunction '" + name + "': file '" + filepath +
                                 "' has fewer than 2 data rows");
    }

    return TableFunction(name, std::move(xs), std::move(ys), index_name, method);
}

} // namespace bngsim
