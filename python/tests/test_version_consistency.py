"""Version consistency cross-check (issue #31).

After PR for #31, ``pyproject.toml`` is the only place a literal
``"X.Y.Z"`` version string lives. Every other anchor (Python
``__version__``, the C extension ``__version__``, the HDF5 attribute
written by ``Result.save``, and the ``project(... VERSION ...)`` line
in ``CMakeLists.txt``) is derived from it.

These tests act as a regression net: if a future contributor
re-introduces a hardcoded literal in any of those files, the test
fails with a clear pointer at the offending file.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import bngsim
import pytest


def _bngsim_source_root() -> Path | None:
    """Locate the ``bngsim/`` source tree, or ``None`` if not findable.

    ``run_tests.sh`` copies test files to a temp directory, so we
    can't rely on a fixed ``__file__`` walk-up. Prefer the
    ``BNGSIM_TEST_DATA`` env var set by the rig (its parent is the
    source root), then ``BNGSIM_SOURCE_ROOT`` for direct overrides,
    then a sibling-pyproject walk from this file as a final
    fallback for in-place pytest runs.
    """
    candidates: list[Path] = []
    env_data = os.environ.get("BNGSIM_TEST_DATA")
    if env_data:
        # BNGSIM_TEST_DATA points at bngsim/tests/data; walk up.
        candidates.extend(Path(env_data).resolve().parents)
    env_root = os.environ.get("BNGSIM_SOURCE_ROOT")
    if env_root:
        candidates.append(Path(env_root).resolve())
    candidates.extend(Path(__file__).resolve().parents)
    for candidate in candidates:
        py = candidate / "pyproject.toml"
        if py.is_file() and 'name = "bngsim"' in py.read_text():
            return candidate
    return None


@pytest.fixture(scope="module")
def src_root() -> Path:
    root = _bngsim_source_root()
    if root is None:
        pytest.skip("Cannot locate bngsim source root for version-consistency check.")
    return root


@pytest.fixture(scope="module")
def pyproject_version(src_root: Path) -> str:
    match = re.search(
        r'^version\s*=\s*"([^"]+)"',
        (src_root / "pyproject.toml").read_text(),
        re.MULTILINE,
    )
    assert match is not None, f"No version line in {src_root / 'pyproject.toml'}"
    return match.group(1)


def test_python_version_matches_pyproject(pyproject_version: str) -> None:
    assert bngsim.__version__ == pyproject_version


def test_c_extension_version_matches_pyproject(pyproject_version: str) -> None:
    from bngsim import _bngsim_core

    assert _bngsim_core.__version__ == pyproject_version


def test_init_py_has_no_version_literal(src_root: Path) -> None:
    """``__init__.py`` must not contain a hardcoded ``__version__ = "X.Y.Z"``."""
    init_py = src_root / "python" / "bngsim" / "__init__.py"
    assert not re.search(r'__version__\s*=\s*"', init_py.read_text()), (
        f"{init_py} contains a literal __version__ assignment; "
        "it should import from bngsim._version instead (issue #31)."
    )


def test_result_py_has_no_version_literal(src_root: Path) -> None:
    """``_result.py`` must not write a hardcoded version string to HDF5."""
    result_py = src_root / "python" / "bngsim" / "_result.py"
    bad = re.search(r'attrs\[\s*"bngsim_version"\s*\]\s*=\s*"', result_py.read_text())
    assert not bad, (
        f"{result_py} writes a literal version string to HDF5; "
        "it should use bngsim._version.__version__ (issue #31)."
    )


def test_core_cpp_has_no_version_literal(src_root: Path) -> None:
    """``_bngsim_core.cpp`` must source ``__version__`` from the CMake compile define."""
    core_cpp = src_root / "src" / "_bngsim_core.cpp"
    bad = re.search(r'attr\(\s*"__version__"\s*\)\s*=\s*"\d', core_cpp.read_text())
    assert not bad, (
        f"{core_cpp} contains a literal __version__ string; "
        "it should use BNGSIM_VERSION_STR from CMake (issue #31)."
    )


def test_cmakelists_has_no_version_literal(src_root: Path) -> None:
    """``CMakeLists.txt`` must not embed an ``X.Y.Z`` literal in ``project(VERSION ...)``."""
    cmakelists = src_root / "CMakeLists.txt"
    bad = re.search(r"project\([^)]*VERSION\s+\d+\.\d+\.\d+", cmakelists.read_text())
    assert not bad, (
        f"{cmakelists} contains a literal version in project(VERSION ...); "
        "it should use ${BNGSIM_VERSION} parsed from pyproject.toml (issue #31)."
    )


def test_pyproject_is_the_only_anchor(src_root: Path, pyproject_version: str) -> None:
    """End-to-end: the four derived anchors must not embed the literal version.

    ``pyproject.toml`` is the single allowed home; ``CHANGELOG.md`` and
    ``uv.lock`` legitimately mention versions too but are not authoritative.
    """
    quoted = f'"{pyproject_version}"'
    derived_anchors = (
        src_root / "python" / "bngsim" / "__init__.py",
        src_root / "python" / "bngsim" / "_result.py",
        src_root / "src" / "_bngsim_core.cpp",
        src_root / "CMakeLists.txt",
    )
    offenders = [
        str(path.relative_to(src_root)) for path in derived_anchors if quoted in path.read_text()
    ]
    assert not offenders, (
        f"Version literal {quoted} found in non-pyproject anchors: {offenders}. "
        "pyproject.toml must remain the single source of truth (issue #31)."
    )


def test_cmake_project_version_picks_up_pyproject(src_root: Path, pyproject_version: str) -> None:
    """When a build tree exists, its CMakeCache should reflect pyproject's version."""
    build = src_root / "build"
    if not build.is_dir():
        pytest.skip("No CMake build dir; skipping CMakeCache cross-check.")
    caches = list(build.rglob("CMakeCache.txt"))
    if not caches:
        pytest.skip("No CMakeCache.txt under build/; skipping.")
    # Multiple wheel-tag build dirs may coexist; any one matching is enough.
    expected = f"CMAKE_PROJECT_VERSION:STATIC={pyproject_version}"
    matched = any(expected in cache.read_text(errors="replace") for cache in caches)
    assert matched, (
        f"No CMakeCache.txt under build/ has {expected}. "
        "Reconfigure CMake after bumping pyproject.toml."
    )
