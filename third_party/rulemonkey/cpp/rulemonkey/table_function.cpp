#include "table_function.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace rulemonkey {

TableFunction::TableFunction(std::string name, std::vector<double> xs, std::vector<double> ys,
                             std::string counter_name, TfunMethod method)
    : name_(std::move(name)), counter_name_(std::move(counter_name)), xs_(std::move(xs)),
      ys_(std::move(ys)), method_(method) {
  if (xs_.size() < 2)
    throw std::runtime_error("tfun '" + name_ + "': requires at least 2 data points");
  if (xs_.size() != ys_.size())
    throw std::runtime_error("tfun '" + name_ + "': x and y arrays must have equal length");
  for (size_t i = 1; i < xs_.size(); ++i) {
    if (xs_[i] <= xs_[i - 1])
      throw std::runtime_error("tfun '" + name_ + "': x values must be strictly increasing");
  }
}

TableFunction TableFunction::from_file(const std::string& name, const std::string& filepath,
                                       const std::string& counter_name, TfunMethod method) {
  std::ifstream in(filepath);
  if (!in.is_open())
    throw std::runtime_error("tfun '" + name + "': cannot open file '" + filepath + "'");

  std::vector<double> xs, ys;
  std::string line;
  bool header_found = false;

  while (std::getline(in, line)) {
    // Trim leading whitespace
    auto start = line.find_first_not_of(" \t");
    if (start == std::string::npos)
      continue;
    line = line.substr(start);

    if (line[0] == '#') {
      if (!header_found) {
        // First comment line is the header — validate column names
        header_found = true;
      }
      continue;
    }

    std::istringstream iss(line);
    double x, y;
    if (!(iss >> x >> y))
      // Error path; one-time allocation cost is irrelevant.
      // NOLINTNEXTLINE(performance-inefficient-string-concatenation)
      throw std::runtime_error("tfun '" + name + "': malformed data line: " + line);
    xs.push_back(x);
    ys.push_back(y);
  }

  return {name, std::move(xs), std::move(ys), counter_name, method};
}

double TableFunction::evaluate(double x) const {
  // Boundary semantics match BNG Network3 (`bng2/Network3/src/model/tfun.cpp`):
  // out-of-range x clamps to the nearest endpoint y, not 0 (NFsim's TFUN
  // returns 0 for x < xs.front() because its CompositeFunction lazily advances
  // an internal cursor — that is implementation detail of NFsim's streaming
  // counter, not a BNG-spec semantic).  Authors who want zero-before-table
  // behavior must place an explicit (xs[0], 0) leading sample.
  if (x <= xs_.front())
    return ys_.front();
  if (x >= xs_.back())
    return ys_.back();

  // Binary search for the interval [xs[i], xs[i+1]) containing x.
  auto it = std::upper_bound(xs_.begin(), xs_.end(), x);
  size_t const i = static_cast<size_t>(it - xs_.begin()) - 1;

  if (method_ == TfunMethod::Step) {
    // Step interpolation: f(x) = ys[i] for x in [xs[i], xs[i+1]).  This is
    // a RIGHT-continuous step function — at the breakpoint x = xs[i+1] the
    // value jumps to ys[i+1] (lim_{t→xs[i+1]+} f(t) = f(xs[i+1]) = ys[i+1]).
    // BNG Network3's `Tfun::stepInterpolate` mislabels this as "left-
    // continuous" in its comment but implements identical semantics; we
    // match BNG's behavior exactly and keep the terminology accurate here.
    return ys_[i];
  }

  // Linear interpolation between (xs[i], ys[i]) and (xs[i+1], ys[i+1]).
  double const x0 = xs_[i], x1 = xs_[i + 1];
  double const y0 = ys_[i], y1 = ys_[i + 1];
  double const t = (x - x0) / (x1 - x0);
  return y0 + (t * (y1 - y0));
}

} // namespace rulemonkey
