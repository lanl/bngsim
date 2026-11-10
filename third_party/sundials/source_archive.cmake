# Where a build takes the pinned SUNDIALS source archive from (lanl/bngsim#1607).
#
# By default the archive is downloaded from VENDOR.json's release URL. When the
# environment variable BNGSIM_SUNDIALS_ARCHIVE names a file, the build reads that
# file instead and downloads nothing. CI sets it to a copy kept in the Actions
# cache, so a GitHub release-download outage no longer fails every build. A
# developer building without a network can set it to a downloaded copy.
#
# The file must be the archive itself, not an unpacked source tree. FetchContent
# checks it against URL_HASH (VENDOR.json's sha256) and unpacks and patches it
# exactly as it would the download, so the #1301 carry is applied either way. A
# FETCHCONTENT_SOURCE_DIR_SUNDIALS source tree skips that patch step.
#
# The name must be an absolute path: a relative one would be read against
# whatever directory the build backend configures in. A name that is not
# absolute, or not a file, falls back to the download, with a warning, rather
# than stopping the configure.
#
# python/tests/test_sundials_source_archive.py runs this under cmake -P.

function(bngsim_sundials_source_url out_var release_url)
    set(_url "${release_url}")
    if(NOT "$ENV{BNGSIM_SUNDIALS_ARCHIVE}" STREQUAL "")
        file(TO_CMAKE_PATH "$ENV{BNGSIM_SUNDIALS_ARCHIVE}" _local)
        if(NOT IS_ABSOLUTE "${_local}")
            message(WARNING
                "BNGSIM_SUNDIALS_ARCHIVE names ${_local}, which is not an absolute "
                "path. Downloading ${release_url} instead.")
        elseif(EXISTS "${_local}" AND NOT IS_DIRECTORY "${_local}")
            set(_url "${_local}")
        else()
            message(WARNING
                "BNGSIM_SUNDIALS_ARCHIVE names ${_local}, which is not a file. "
                "Downloading ${release_url} instead.")
        endif()
    endif()
    set(${out_var} "${_url}" PARENT_SCOPE)
endfunction()
