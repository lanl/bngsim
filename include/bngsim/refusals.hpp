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

// An ODE run over a span CVODE cannot step through, refused before it starts
// (issue #1838): a time 2^512 or more from t = 0, or a run that long. Only the
// ODE engine refuses it, so the engine does, not the checks of a span that
// Python makes for every method. A std::invalid_argument, which the pybind11
// layer maps to `bngsim.UsageError`: a ValueError, as before it had a type, and
// a BngsimError with a code, as the refusals of a span Python raises are.
class TimeSpanOutOfReach : public std::invalid_argument {
  public:
    explicit TimeSpanOutOfReach(const std::string &what) : std::invalid_argument(what) {}
};

} // namespace bngsim
