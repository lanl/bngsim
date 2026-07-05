// bngsim/include/bngsim/bngsim.hpp — Top-level convenience header
//
// Usage: #include <bngsim/bngsim.hpp>

#pragma once

#include "bngsim/expression.hpp"
#include "bngsim/model.hpp"
#include "bngsim/model_builder.hpp"
#include "bngsim/net_file_loader.hpp"
#include "bngsim/result.hpp"
#include "bngsim/simulator.hpp"
#include "bngsim/steady_state.hpp"
#include "bngsim/table_function.hpp"
#include "bngsim/types.hpp"

#ifdef BNGSIM_HAS_NFSIM
#include "bngsim/nfsim_simulator.hpp"
#endif

#ifdef BNGSIM_HAS_RULEMONKEY
#include "bngsim/rulemonkey_simulator.hpp"
#endif
