#pragma once

#include <string>
#include <vector>

namespace rulemonkey {

enum class TfunMethod { Linear, Step };

class TableFunction {
public:
  TableFunction(std::string name, std::vector<double> xs, std::vector<double> ys,
                std::string counter_name, TfunMethod method = TfunMethod::Linear);

  // Load from a .tfun file. Validates header against name and counter_name.
  static TableFunction from_file(const std::string& name, const std::string& filepath,
                                 const std::string& counter_name,
                                 TfunMethod method = TfunMethod::Linear);

  double evaluate(double x) const;

  const std::string& name() const { return name_; }
  const std::string& counter_name() const { return counter_name_; }
  TfunMethod method() const { return method_; }
  size_t size() const { return xs_.size(); }

private:
  std::string name_;
  std::string counter_name_;
  std::vector<double> xs_;
  std::vector<double> ys_;
  TfunMethod method_;
};

} // namespace rulemonkey
