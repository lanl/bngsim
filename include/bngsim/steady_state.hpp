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
// method="integration" (default): CVODE parity early-stop only.
// method="newton": two-tier integrate-first solver (GH #27). A CVODE burst
//                carries the state into the physical root's basin, then KINSOL
//                polishes; the polish is accepted only once it is seed-stable
//                (agrees across two successively tighter bursts), otherwise
//                integration continues. Correct on multi-root and NaN-prone
//                models where seeding Newton at the raw IC returned spurious /
//                non-finite roots — but since GH #27 made it integrate first,
//                the polish is strictly extra work on top of the integration
//                path, and GH #28 measured it as a 1.4-3.9x net cost on every
//                benchmarked model. It buys a much tighter residual (~1e-13
//                vs ~1e-9), which is why it stays available.
// method="kinsol": accepted alias for "newton".
SteadyStateResult find_steady_state(NetworkModel &model, const SteadyStateOptions &opts = {});

} // namespace bngsim
