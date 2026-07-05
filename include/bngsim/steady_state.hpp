// bngsim/include/bngsim/steady_state.hpp -- steady-state solver
//
// All integrate-to-steady-state paths use the BNG2.pl parity criterion
// ||f(y)||_2 / n_species < tol (run_network -c), the same rule applied by
// Simulator.run(steady_state=True).
//
#pragma once

#include "bngsim/model.hpp"
#include "bngsim/types.hpp"

namespace bngsim {

// Find the steady state of the ODE system f(y) = 0.
//
// method="newton" (default): KINSOL Newton solver; on non-convergence it
//                falls back EXPLICITLY to the parity integration path.
// method="integration": CVODE parity early-stop only.
// method="kinsol": accepted alias for "newton".
SteadyStateResult find_steady_state(NetworkModel &model, const SteadyStateOptions &opts = {});

} // namespace bngsim
