#pragma once

#include <stdexcept>
#include <string>

namespace bngsim {

// A run bngsim refuses by name because the model cannot supply what an option
// asks for: force_sparse_linear_solver on a model with no sparse Jacobian to hand
// KLU (issue #1313). Distinct from std::runtime_error so the pybind11 layer maps
// it to `bngsim.SparseJacobianUnavailableError`, a SimulationError of its own: a
// caller tells the refusal from a solver failure by its type, not its text.
class SparseJacobianUnavailable : public std::runtime_error {
  public:
    explicit SparseJacobianUnavailable(const std::string &what) : std::runtime_error(what) {}
};

} // namespace bngsim
