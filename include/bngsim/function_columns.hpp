// bngsim/include/bngsim/function_columns.hpp
//
// Helpers for the BNG2.pl .gdat/.scan function-column convention, shared by the
// Result serializer (result.cpp) and the Python bindings (_bngsim_core.cpp).
//
// BNG2.pl synthesises internal rate-law functions (`_rateLaw1`, `_rateLaw2`, …)
// into BNG-XML's <ListOfFunctions> for run_network/NFsim, and its own
// .gdat/.scan writer filters them out by default and decorates user-named
// functions with a trailing "()" (e.g. `kf_BSA()`).
//
// bngsim deliberately normalises this (issue #58): function headers are always
// bare (no "()"), identical across every simulation method, and the synthetic
// `_rateLawN` columns are omitted by default. The `print_functions` flag opts
// the user-named functions back in; the `print_rate_laws` flag additionally
// opts the `_rateLawN` columns in (still bare). These helpers select the
// columns; the bare name is the header label.

#pragma once

#include <string>
#include <vector>

namespace bngsim {

// True for auto-generated rate-law function names: `_rateLaw` followed by one
// or more digits and nothing else (e.g. `_rateLaw1`, `_rateLaw42`).
inline bool is_auto_rate_law(const std::string &name) {
    static const std::string prefix = "_rateLaw";
    if (name.size() <= prefix.size() || name.compare(0, prefix.size(), prefix) != 0) {
        return false;
    }
    for (size_t i = prefix.size(); i < name.size(); ++i) {
        if (name[i] < '0' || name[i] > '9') {
            return false;
        }
    }
    return true;
}

// Column indices of the user-named (non-_rateLawN) functions, in order.
inline std::vector<size_t> user_function_indices(const std::vector<std::string> &names) {
    std::vector<size_t> keep;
    keep.reserve(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        if (!is_auto_rate_law(names[i])) {
            keep.push_back(i);
        }
    }
    return keep;
}

// Column indices selected for .gdat/.scan output, in declared order: the
// user-named functions when ``print_functions``, plus the auto-generated
// ``_rateLawN`` columns when ``print_rate_laws``. Both default off → no
// function columns at all (observables-only output).
inline std::vector<size_t> gdat_function_indices(const std::vector<std::string> &names,
                                                 bool print_functions, bool print_rate_laws) {
    std::vector<size_t> keep;
    keep.reserve(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        const bool is_rl = is_auto_rate_law(names[i]);
        if ((is_rl && print_rate_laws) || (!is_rl && print_functions)) {
            keep.push_back(i);
        }
    }
    return keep;
}

// True for bngsim-internal scaffolding observables. The .net loader's
// network-rewrite pass synthesises per-reactant single-species observables to
// back legacy Sat/Hill functional rate laws (named `__bngsim_net_rewrite_obs_*`,
// see net_file_loader.cpp). They must live in the model so the rewritten
// functional rate-law expressions can reference them by name, but they are
// internal and must never surface in user-facing .gdat output or the
// observable_{names,data,n_observables} API — run_network never emits them
// (issue #61). The `__bngsim_` prefix is a reserved namespace no user-declared
// observable can occupy.
inline bool is_internal_observable(const std::string &name) {
    static const std::string prefix = "__bngsim_";
    return name.size() > prefix.size() && name.compare(0, prefix.size(), prefix) == 0;
}

// Column indices of the user-facing (non-internal) observables, in order.
inline std::vector<size_t> public_observable_indices(const std::vector<std::string> &names) {
    std::vector<size_t> keep;
    keep.reserve(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
        if (!is_internal_observable(names[i])) {
            keep.push_back(i);
        }
    }
    return keep;
}

} // namespace bngsim
