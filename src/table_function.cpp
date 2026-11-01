// bngsim/src/table_function.cpp — Table function (tfun) implementation
//
// Piecewise-linear or step interpolation of tabular data.
// Binary search + interpolation with constant extrapolation beyond endpoints.
// Supports both file-based tfuns and inline tfun array syntax.

#include "bngsim/table_function.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
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

    // A NaN x value slips through the monotonicity test below (both `<=`
    // comparisons are false) and then breaks the binary search in evaluate_at()
    // the same way a NaN index does (issue #778), so refuse it by position.
    for (size_t i = 0; i < xs_.size(); ++i) {
        if (std::isnan(xs_[i])) {
            throw std::runtime_error("TableFunction '" + name_ + "': x[" + std::to_string(i) +
                                     "] is NaN");
        }
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
    // A NaN index has no place in the table (issue #778). Every comparison with
    // NaN is false, so it fell through both endpoint tests and upper_bound()
    // returned end(): a step table answered its LAST value — finite, plausible
    // and wrong, with no warning — and a linear table read xs_[n]/ys_[n], one
    // past the end. Propagate it instead, so the RHS non-finite guards refuse
    // the run the way they refuse any other uncomputable rate (#580).
    if (std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();

    // Constant extrapolation beyond endpoints
    if (x <= xs_.front())
        return ys_.front();
    if (x >= xs_.back())
        return ys_.back();

    // Binary search for the interval containing x
    // Find the first element > x, then back up one. With x strictly inside
    // (front, back) that is always in [0, n-2]; the clamp keeps the linear
    // branch's i+1 in bounds even if that invariant is ever broken.
    auto it = std::upper_bound(xs_.begin(), xs_.end(), x);
    const int last = static_cast<int>(xs_.size()) - 2;
    int i = std::clamp(static_cast<int>(it - xs_.begin()) - 1, 0, last);

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

double TableFunction::slope_at(double x) const {
    if (std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (method_ == InterpolationMethod::Step || x <= xs_.front() || x >= xs_.back())
        return 0.0;
    // evaluate_at's interval, found as it finds it.
    auto it = std::upper_bound(xs_.begin(), xs_.end(), x);
    const int last = static_cast<int>(xs_.size()) - 2;
    const int i = std::clamp(static_cast<int>(it - xs_.begin()) - 1, 0, last);
    return (ys_[i + 1] - ys_[i]) / (xs_[i + 1] - xs_[i]);
}

double TableFunction::left_slope_at(double x) const {
    if (std::isnan(x))
        return std::numeric_limits<double>::quiet_NaN();
    if (method_ == InterpolationMethod::Step || x <= xs_.front() || x > xs_.back())
        return 0.0;
    // The first knot at or above x closes the interval (x0, x1] that holds it.
    auto it = std::lower_bound(xs_.begin(), xs_.end(), x);
    const int i =
        std::clamp(static_cast<int>(it - xs_.begin()), 1, static_cast<int>(xs_.size()) - 1);
    return (ys_[i] - ys_[i - 1]) / (xs_[i] - xs_[i - 1]);
}

bool TableFunction::is_knot(double x) const {
    return std::binary_search(xs_.begin(), xs_.end(), x);
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
                                       const std::string &header_name,
                                       const std::string &also_header) {
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
                if (strip_paren_suffix(col2) != strip_paren_suffix(expected_header) &&
                    (also_header.empty() ||
                     strip_paren_suffix(col2) != strip_paren_suffix(also_header))) {
                    throw std::runtime_error("TableFunction '" + name + "': header column 2 '" +
                                             col2 + "' does not match expected function name '" +
                                             expected_header + "'" +
                                             (also_header.empty() || also_header == expected_header
                                                  ? std::string()
                                                  : " or '" + also_header + "'"));
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

void TableFunction::take_data_of(const TableFunction &other) {
    xs_ = other.xs_;
    ys_ = other.ys_;
    method_ = other.method_;
}

} // namespace bngsim
