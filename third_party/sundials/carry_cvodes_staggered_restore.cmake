# BNGsim's one local carry in the fetched SUNDIALS source (lanl/bngsim#1301).
#
# Run as the PATCH_COMMAND of the SUNDIALS FetchContent declaration in
# CMakeLists.txt:  cmake -DSUNDIALS_SOURCE_DIR=<dir> -P <this file>
#
# In cvStep's staggered sensitivity corrector, CVODES evaluates f at the
# converged state, and a recoverable failure there ("treat it as a convergence
# failure and attempt the step again", its comment says) goes back round the
# loop with neither cvRestore nor a smaller step. The next attempt predicts
# again from the advanced tn: tn moves by 2h, the first h of it is crossed by
# the uncorrected predictor, and the error test and the root search see only
# the second. On MODEL2001200002 (Mpekris 2017) with sensitivities that left
# the state 9e-3 and the sensitivities 1.4e-2 off, with no error, and output
# times outside the last step that CVode returned anyway, with the state
# unread. The carry sends the failure through cvHandleNFlag, as every other
# recoverable failure in cvStep goes. Present in SUNDIALS 7.2.1 through 7.9.0.
#
# The step's limit is checked against the sensitivity corrector's count (ncfS),
# which a step keeps across its attempts. The state's count (ncf) is reset to 0
# just above the call on every attempt, so a refusal that a smaller step does
# not cure never reached maxncf with it: the step shrank to ~1e-16 forever
# (MODEL2001140002, which upstream hangs on as well). As f is the state RHS, the
# failure is counted in cv_ncfn (CVodeGetNumStepSolveFails), as the upstream PR
# counts it. r2 counted it in cv_ncfnS, which CV_STAGGERED1 (BNGsim uses
# CV_STAGGERED) overwrites with the sum of its per-parameter counts at the end of
# each CVode call. BNGsim reads neither count, and neither steers the step, so r2
# and r3 give the same runs.
#
# Idempotent: a source that already carries this revision's marker is left as
# it is. A source with an older revision, or with neither the marker nor the
# upstream text, is an error, so neither a stale build tree nor a SUNDIALS
# refresh keeps the wrong code silently.

if(NOT DEFINED SUNDIALS_SOURCE_DIR)
    message(FATAL_ERROR "carry_cvodes_staggered_restore.cmake: pass -DSUNDIALS_SOURCE_DIR=<dir>")
endif()

set(_file "${SUNDIALS_SOURCE_DIR}/src/cvodes/cvodes.c")
if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "carry_cvodes_staggered_restore.cmake: no ${_file}")
endif()
file(READ "${_file}" _text)

set(_marker "BNGSIM LOCAL CARRY (lanl/bngsim#1301, r3)")
string(FIND "${_text}" "${_marker}" _at)
if(NOT _at EQUAL -1)
    return()
endif()
string(FIND "${_text}" "BNGSIM LOCAL CARRY (lanl/bngsim#1301" _older)
if(NOT _older EQUAL -1)
    message(FATAL_ERROR
        "carry_cvodes_staggered_restore.cmake: ${_file} holds an older revision of the "
        "lanl/bngsim#1301 carry. A build tree's fetch: delete its _deps/sundials-* "
        "directories and configure again, so FetchContent fetches the source afresh. "
        "A FETCHCONTENT_SOURCE_DIR_SUNDIALS copy: restore src/cvodes/cvodes.c from a "
        "pristine SUNDIALS of the pinned version, then run this script again.")
endif()

set(_upstream [==[
      if (retval < 0) { return (CV_RHSFUNC_FAIL); }
      if (retval > 0)
      {
        nflag = PREV_CONV_FAIL;
        continue;
      }
]==])

set(_carried [==[
      if (retval < 0) { return (CV_RHSFUNC_FAIL); }
      if (retval > 0)
      {
        /* BNGSIM LOCAL CARRY (lanl/bngsim#1301, r3): restore the prediction
         * and cut the step, as every other recoverable failure in cvStep does.
         * Upstream continues with neither, so the next attempt predicts again
         * from the advanced tn. The limit is checked against ncfS, which is not
         * reset between attempts as ncf is above, so maxncf ends a refusal no
         * smaller step cures. As f is the state RHS, the failure is counted in
         * ncfn, as LLNL/sundials#1073 counts it.
         * BNGSIM_SUNDIALS_STAGGERED_RESTORE=0 takes the upstream path, for an
         * A/B only. */
        const char* bngsim_restore = getenv("BNGSIM_SUNDIALS_STAGGERED_RESTORE");
        if (bngsim_restore == NULL || bngsim_restore[0] != '0' ||
            bngsim_restore[1] != 0)
        {
          nflag = RHSFUNC_RECVR;
          kflag = cvHandleNFlag(cv_mem, &nflag, saved_t, &ncfS,
                                &(cv_mem->cv_ncfn));
          if (kflag == PREDICT_AGAIN) { continue; }
          return (kflag);
        }
        nflag = PREV_CONV_FAIL;
        continue;
      }
]==])

string(FIND "${_text}" "${_upstream}" _first)
string(FIND "${_text}" "${_upstream}" _last REVERSE)
if(_first EQUAL -1 OR NOT _first EQUAL _last)
    message(FATAL_ERROR
        "carry_cvodes_staggered_restore.cmake: ${_file} does not hold the upstream "
        "staggered-corrector block exactly once, so the lanl/bngsim#1301 carry cannot "
        "be applied. A SUNDIALS refresh moved it: re-derive the carry against the new "
        "cvStep (see third_party/sundials/VENDOR.json, local_carries).")
endif()
string(REPLACE "${_upstream}" "${_carried}" _text "${_text}")
file(WRITE "${_file}" "${_text}")
message(STATUS "SUNDIALS: applied the lanl/bngsim#1301 carry to src/cvodes/cvodes.c")
