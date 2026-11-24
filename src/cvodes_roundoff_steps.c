/* bngsim/src/cvodes_roundoff_steps.c — CVODES' count of the steps too small to
 * move its clock (issue #1837).
 *
 * Before each step CVode checks whether t + h == t in double precision, and
 * where it is, counts the step (cv_nhnil), prints the "t + h = t on the next
 * step" warning for the first mxhnil (10) of them, and takes the step. Such a
 * step moves the state by h and the clock not at all, so every row after it can
 * be wrong with no solver error. CVODES keeps the count and has no function to
 * read it, so this one file is compiled against the CVODES source bngsim fetches
 * and reads it there. CVodeInit and CVodeReInit zero it, as they zero the
 * counters CVodeGetNumSteps and the rest read.
 *
 * C, not C++, because the private headers are C. Built only where the fetched
 * SUNDIALS source is there (CMakeLists.txt); a system SUNDIALS build counts no
 * such step, and its configure says so. */

#include "cvodes_impl.h"

long bngsim_cvodes_roundoff_steps(void *cvode_mem) {
    if (cvode_mem == NULL)
        return 0;
    return (long)((CVodeMem)cvode_mem)->cv_nhnil;
}
