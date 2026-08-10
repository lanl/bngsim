# Point find_package(pybind11) at the interpreter this configure is building for.
#
# pybind11 ships its CMake package inside site-packages, so `find_package(pybind11
# CONFIG)` cannot find it until something names an interpreter. This module is that
# something: it walks a candidate list, asks each one for `pybind11.get_cmake_dir()`,
# and pins `pybind11_DIR` to the first usable answer.
#
# The order of that list decides which pybind11 compiles the extension, and getting
# it wrong is silent — the wheel records nothing about the pybind11 that built it,
# and any recent pybind11 produces a module that imports and works.
#
# The rule, in the order applied:
#
# 1. `$BNGSIM_PYTHON_EXECUTABLE` — the deliberate override. It stays ahead of
#    everything precisely because it is the escape hatch for a machine where the
#    rest of this list picks wrong.
# 2. `Python_EXECUTABLE` (then the `Python3_`/`PYTHON_` spellings) — the
#    interpreter this configure actually targets. scikit-build-core sets it for
#    every wheel and editable build, and `scripts/rebuild_editable.py` pins it
#    explicitly; in all of those the answer is not a guess. It was absent from
#    this list until GH #288, which is what let a checkout-local `.venv` supply
#    pybind11 to wheel builds targeting a completely different interpreter —
#    including isolated builds whose own `[build-system] requires` had already
#    resolved and installed a newer one.
# 3. `$VIRTUAL_ENV`, then a `.venv` beside the source, then `python3` on PATH —
#    the fallbacks that make the editable rebuild path work "after the
#    build-isolation venv is gone" (GH #23, #229). That is still the right rule
#    for a build directory re-driven later by hand, and it is why entry 2 is a
#    prepend rather than a replacement: when `Python_EXECUTABLE` names a deleted
#    build env, or names a live interpreter with no pybind11 in it, the walk
#    falls straight through to these.
#
# The deleted-build-env case is handled before the walk starts, by dropping cache
# entries whose paths no longer exist — otherwise a phantom `Python_EXECUTABLE`
# would now be consulted first and resolve nothing, and a phantom `pybind11_DIR`
# would be trusted and never re-resolved.
#
# This lives in its own file so the ordering can be exercised directly against
# fake interpreters, with no project configure — see
# python/tests/test_cmake_pybind11_resolution.py.

# Note on the repeated `Python_EXECUTABLE Python3_EXECUTABLE PYTHON_EXECUTABLE`
# lists below — those are the three names CMake/FindPython use for "the
# interpreter this build targets", most authoritative first. They are spelled out
# in each function rather than held in a shared variable because a CMake function
# body resolves names through its *caller's* scope: a file-scope list would be
# visible only to callers that happen to share the scope this module was
# included into.

# True iff ${dir} is a directory a pybind11 CONFIG search would accept.
function(_bngsim_pybind11_config_dir dir out_var)
    if(dir AND (EXISTS "${dir}/pybind11Config.cmake" OR EXISTS "${dir}/pybind11-config.cmake"))
        set(${out_var} TRUE PARENT_SCOPE)
    else()
        set(${out_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

# Drop cached interpreter / pybind11 paths that no longer exist on disk.
#
# `editable.rebuild = false` (pyproject.toml) means the build directory is
# re-driven later by scripts/rebuild_editable.py, long after the uv or pip
# build-isolation venv that configured it has been deleted. Everything cached
# from inside that venv is a phantom by then; leaving it in place makes the
# rebuild point at nothing. See GH #23.
function(bngsim_drop_phantom_python_cache)
    foreach(_var IN ITEMS Python_EXECUTABLE Python3_EXECUTABLE PYTHON_EXECUTABLE)
        if(${_var} AND NOT EXISTS "${${_var}}")
            unset(${_var} CACHE)
            unset(_${_var} CACHE)
        endif()
    endforeach()
    _bngsim_pybind11_config_dir("${pybind11_DIR}" _usable)
    if(pybind11_DIR AND NOT _usable)
        unset(pybind11_DIR CACHE)
    endif()
endfunction()

# Resolve and pin pybind11_DIR. No-op when it is already set to a usable
# directory, so an explicit -Dpybind11_DIR= always wins outright.
function(bngsim_resolve_pybind11_dir)
    bngsim_drop_phantom_python_cache()
    if(pybind11_DIR)
        return()
    endif()

    set(_candidates "")
    if(DEFINED ENV{BNGSIM_PYTHON_EXECUTABLE})
        list(APPEND _candidates "$ENV{BNGSIM_PYTHON_EXECUTABLE}")
    endif()
    foreach(_var IN ITEMS Python_EXECUTABLE Python3_EXECUTABLE PYTHON_EXECUTABLE)
        if(${_var})
            list(APPEND _candidates "${${_var}}")
        endif()
    endforeach()
    # $VIRTUAL_ENV is unset when `.venv/bin/python` is invoked by path, hence the
    # source-relative entries; FindPython resolves a venv symlink to the base
    # interpreter, which lacks the venv's site-packages, hence not relying on it.
    if(DEFINED ENV{VIRTUAL_ENV})
        list(APPEND _candidates
            "$ENV{VIRTUAL_ENV}/bin/python"
            "$ENV{VIRTUAL_ENV}/Scripts/python.exe"
        )
    endif()
    list(APPEND _candidates
        "${CMAKE_SOURCE_DIR}/.venv/bin/python"
        "${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe"
        "${CMAKE_SOURCE_DIR}/../.venv/bin/python"
        "${CMAKE_SOURCE_DIR}/../.venv/Scripts/python.exe"
    )
    find_program(_bngsim_py_path NAMES python3 python)
    if(_bngsim_py_path)
        list(APPEND _candidates "${_bngsim_py_path}")
    endif()

    foreach(_py IN LISTS _candidates)
        if(NOT EXISTS "${_py}")
            continue()
        endif()
        execute_process(
            COMMAND "${_py}" -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE _dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _rc
            ERROR_QUIET
        )
        if(NOT _rc EQUAL 0)
            continue()
        endif()
        # A directory with no config file in it would turn a clear "could not
        # find pybind11" into a stranger error about a bad pybind11_DIR, and
        # would stop the walk at an interpreter that cannot actually supply it.
        _bngsim_pybind11_config_dir("${_dir}" _usable)
        if(_usable)
            set(pybind11_DIR "${_dir}" CACHE PATH "pybind11 cmake dir" FORCE)
            message(STATUS "BNGsim: pybind11 resolved via ${_py}")
            return()
        endif()
    endforeach()

    # Nothing answered. Deliberately not fatal: find_package's own CONFIG search
    # still runs next and a system-wide pybind11 is a legitimate way to build
    # (GH #229) — it just has to be the last resort rather than a silent first.
    message(STATUS
        "BNGsim: no interpreter supplied pybind11; leaving discovery to find_package")
endfunction()
