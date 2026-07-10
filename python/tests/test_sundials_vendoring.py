"""File-based guardrails for BNGsim's pinned SUNDIALS fetch metadata."""

from __future__ import annotations

import json
from pathlib import Path

BNGSIM_ROOT = Path(__file__).resolve().parents[2]
SUNDIALS_DIR = BNGSIM_ROOT / "third_party" / "sundials"
SUNDIALS_VENDOR_METADATA = SUNDIALS_DIR / "VENDOR.json"
CMAKE_LISTS = BNGSIM_ROOT / "CMakeLists.txt"

EXPECTED_COMPONENT_PATHS = [
    "CMakeLists.txt",
    "README.md",
    "cmake/SundialsBuildOptionsPre.cmake",
    "cmake/SundialsTPLOptions.cmake",
    "cmake/SundialsSetupCompilers.cmake",
    "cmake/tpl/SundialsPOSIXTimers.cmake",
    "cmake/macros/SundialsAddLibrary.cmake",
    "src/CMakeLists.txt",
    "src/cvodes/CMakeLists.txt",
    "src/kinsol/CMakeLists.txt",
    "src/nvector/serial/CMakeLists.txt",
    "src/sunmatrix/dense/CMakeLists.txt",
    "src/sunmatrix/sparse/CMakeLists.txt",
    "src/sunlinsol/dense/CMakeLists.txt",
    "src/sunlinsol/klu/CMakeLists.txt",
]

EXPECTED_REQUIRED_TEXT_TOKENS = {
    "cmake/SundialsBuildOptionsPre.cmake": [
        "sundials_option(SUNDIALS_BUILD_WITH_MONITORING BOOL",
        "sundials_option(SUNDIALS_BUILD_WITH_PROFILING BOOL",
        "sundials_option(BUILD_CVODES BOOL",
        "sundials_option(BUILD_KINSOL BOOL",
    ],
    "cmake/SundialsTPLOptions.cmake": [
        "sundials_option(ENABLE_KLU BOOL",
    ],
    "cmake/SundialsSetupCompilers.cmake": [
        "if(SUNDIALS_POSIX_TIMERS AND POSIX_TIMERS_NEED_POSIX_C_SOURCE)",
    ],
    "cmake/tpl/SundialsPOSIXTimers.cmake": [
        "set(POSIX_TIMERS_NEED_POSIX_C_SOURCE TRUE)",
        "set(SUNDIALS_POSIX_TIMERS TRUE)",
        "set(SUNDIALS_POSIX_TIMERS FALSE)",
    ],
    "cmake/macros/SundialsAddLibrary.cmake": [
        "add_library(SUNDIALS::${_export_name} ALIAS",
    ],
    "src/CMakeLists.txt": [
        "if(BUILD_CVODES)",
        "if(BUILD_KINSOL)",
    ],
    "src/cvodes/CMakeLists.txt": [
        "sundials_add_library(",
    ],
    "src/kinsol/CMakeLists.txt": [
        "sundials_add_library(",
    ],
    "src/nvector/serial/CMakeLists.txt": [
        "sundials_add_library(",
    ],
    "src/sunmatrix/dense/CMakeLists.txt": [
        "sundials_add_library(",
    ],
    "src/sunmatrix/sparse/CMakeLists.txt": [
        "sundials_add_library(",
    ],
    "src/sunlinsol/dense/CMakeLists.txt": [
        "sundials_add_library(",
    ],
    "src/sunlinsol/klu/CMakeLists.txt": [
        "sundials_add_library(",
    ],
}


def test_sundials_vendor_metadata_shape():
    metadata = json.loads(SUNDIALS_VENDOR_METADATA.read_text())

    assert metadata["source"]["authoritative_repo"] == "https://github.com/LLNL/sundials.git"
    assert metadata["source"]["authoritative_release_tag"] == "v7.2.1"
    assert metadata["source"]["authoritative_release_asset_name"] == "sundials-7.2.1.tar.gz"
    assert metadata["source"]["authoritative_release_asset_url"] == (
        "https://github.com/LLNL/sundials/releases/download/v7.2.1/sundials-7.2.1.tar.gz"
    )
    assert metadata["source"]["authoritative_release_page"] == (
        "https://github.com/LLNL/sundials/releases/tag/v7.2.1"
    )
    assert metadata["source"]["tag_object"] == "35a984a216f6ea1db0315a506114673d90994407"
    assert metadata["source"]["tag_commit"] == "2dcb3e018b4c4cfe824bff09eb52184ed083e368"

    assert metadata["files"]["source_archive"]["asset_name"] == "sundials-7.2.1.tar.gz"
    assert metadata["files"]["source_archive"]["sha256"] == (
        "3781e3f7cdf372ca12f7fbe64f561a8b9a507b8a8b2c4d6ce28d8e4df4befbea"
    )
    assert metadata["files"]["source_archive"]["root_dir"] == "sundials-7.2.1"
    assert metadata["files"]["source_archive"]["detected_version"] == "7.2.1"

    assert metadata["guardrails"]["required_component_paths"] == EXPECTED_COMPONENT_PATHS
    assert metadata["guardrails"]["required_text_tokens"] == EXPECTED_REQUIRED_TEXT_TOKENS
    assert metadata["local_carries"] == []


def test_cmake_reads_pinned_sundials_metadata():
    text = CMAKE_LISTS.read_text()

    assert 'set(BNGSIM_SUNDIALS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/sundials")' in text
    assert 'set(BNGSIM_SUNDIALS_VENDOR_METADATA "${BNGSIM_SUNDIALS_DIR}/VENDOR.json")' in text
    assert "authoritative_release_asset_url" in text
    assert "files source_archive sha256" in text
    assert 'URL_HASH "SHA256=${BNGSIM_SUNDIALS_SOURCE_SHA256}"' in text
    assert "DOWNLOAD_EXTRACT_TIMESTAMP FALSE" in text
    assert "vendor_sundials.py" in text
    assert "SUNDIALS_VENDORING.md" in text
