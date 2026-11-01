# Discard a SUNDIALS fetch that predates the current local carry (lanl/bngsim#1301).
#
# FetchContent runs the carry's PATCH_COMMAND when it fetches the source, and
# again only if that command line changes. The carry's marker revision lives in
# carry_cvodes_staggered_restore.cmake, not on the command line, so a build tree
# fetched under an older revision would keep the older code. CMakeLists.txt
# calls this before FetchContent_MakeAvailable(sundials): a fetched cvodes.c
# without the current marker has the source and FetchContent's record of having
# fetched and patched it removed, and FetchContent then fetches and patches the
# source afresh. CMakeLists.txt names the marker, so a bump changes it and every
# build tree reconfigures and reaches this check.
#
# That record is <base_dir>/sundials-subbuild when FetchContent populates
# through a sub-build (policy CMP0168 OLD, which cmake_minimum_required(3.20)
# gives), and <binary_dir>/CMakeFiles/fc-stamp/sundials and fc-tmp/sundials when
# it populates directly (CMP0168 NEW). Removing the source alone would leave the
# stamps saying it was fetched. <base_dir>/sundials-build is kept: the fresh
# source is newer than its objects, so they are rebuilt.
#
# Left alone: a FETCHCONTENT_SOURCE_DIR_SUNDIALS copy, which is the developer's
# own, and a FETCHCONTENT_FULLY_DISCONNECTED build, which cannot fetch again.
# The marker check after FetchContent_MakeAvailable refuses both.
#
# python/tests/test_sundials_discard_stale_fetch.py runs this under cmake -P.

function(bngsim_discard_stale_sundials_fetch base_dir binary_dir marker own_source_dir fully_disconnected)
    if(own_source_dir OR fully_disconnected)
        return()
    endif()
    set(_cvodes "${base_dir}/sundials-src/src/cvodes/cvodes.c")
    if(NOT EXISTS "${_cvodes}")
        return()
    endif()
    file(READ "${_cvodes}" _text)
    string(FIND "${_text}" "${marker}" _at)
    if(NOT _at EQUAL -1)
        return()
    endif()
    message(STATUS
        "BNGsim: the SUNDIALS fetched into ${base_dir} predates the current "
        "lanl/bngsim#1301 carry; discarding it, so it is fetched and patched again")
    file(REMOVE_RECURSE
        "${base_dir}/sundials-src" "${base_dir}/sundials-subbuild"
        "${binary_dir}/CMakeFiles/fc-stamp/sundials" "${binary_dir}/CMakeFiles/fc-tmp/sundials")
endfunction()
