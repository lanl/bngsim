// bngsim/src/cvodes_roundoff_steps.hpp — how many steps of a CVODES
// integration were too small to move its clock (issue #1837).
//
// CVode takes a step where t + h == t in double precision, warns on standard
// error for the first ten, and goes on: the state moves and the clock does not.
// src/cvodes_roundoff_steps.c reads CVODES' own count of them from the fetched
// source's private header, since CVODES has no function that returns it. Here
// and not under include/, which is installed: what it returns turns on a
// definition private to the bngsim target.

#pragma once

#ifdef BNGSIM_SUNDIALS_COUNTS_ROUNDOFF_STEPS
extern "C" long bngsim_cvodes_roundoff_steps(void *cvode_mem);
#endif

namespace bngsim {

// The steps since the last CVodeInit or CVodeReInit of *cvode_mem* where
// t + h == t. 0 in a build against a system SUNDIALS, which counts none.
inline long cvodes_roundoff_steps(void *cvode_mem) {
#ifdef BNGSIM_SUNDIALS_COUNTS_ROUNDOFF_STEPS
    return bngsim_cvodes_roundoff_steps(cvode_mem);
#else
    (void)cvode_mem;
    return 0;
#endif
}

} // namespace bngsim
