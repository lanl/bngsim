"""A build reads the pinned SUNDIALS archive from a local copy when told to (#1607).

Every source build fetched SUNDIALS from its GitHub release URL, so a
release-download outage failed every CI leg at once. CI now keeps the archive
in the Actions cache and names the copy in BNGSIM_SUNDIALS_ARCHIVE.
third_party/sundials/source_archive.cmake turns that into FetchContent's URL.

These tests run the real module under real cmake. Each pins one rule:

* with the variable unset or empty, the URL is VENDOR.json's release URL;
* naming a file, the URL is that file;
* naming nothing, a directory, or a relative path, falls back to the release
  URL with a warning;
* FetchContent takes the local file with no network: it checks it against
  URL_HASH, unpacks it and runs the patch step, as it would the download. A copy
  whose bytes differ stops the configure;
* CMakeLists.txt hands the module's answer to FetchContent_Declare(sundials),
  next to the pinned URL_HASH.
"""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = REPO_ROOT / "third_party" / "sundials" / "source_archive.cmake"
CMAKE_LISTS = REPO_ROOT / "CMakeLists.txt"

CMAKE = shutil.which("cmake")
RELEASE_URL = "https://github.invalid/LLNL/sundials/releases/download/v0/sundials-0.tar.gz"

pytestmark = [
    pytest.mark.skipif(
        not MODULE.exists(),
        reason="source_archive.cmake is not in this checkout (installed package)",
    ),
    pytest.mark.skipif(CMAKE is None, reason="no CMake on PATH to run the module with"),
]


def _env(archive: str | None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "BNGSIM_SUNDIALS_ARCHIVE"}
    if archive is not None:
        env["BNGSIM_SUNDIALS_ARCHIVE"] = archive
    return env


def _url(tmp_path: Path, archive: str | None) -> tuple[str, str]:
    """The module's answer under ``archive``, and what cmake printed, with its
    whitespace collapsed: cmake wraps a warning's text at any space."""
    out = tmp_path / "url.txt"
    driver = tmp_path / "driver.cmake"
    driver.write_text(
        f'include("{MODULE.as_posix()}")\n'
        f'bngsim_sundials_source_url(_url "{RELEASE_URL}")\n'
        f'file(WRITE "{out.as_posix()}" "${{_url}}")\n'
    )
    proc = subprocess.run(
        [CMAKE, "-P", str(driver)], capture_output=True, text=True, timeout=60, env=_env(archive)
    )
    assert proc.returncode == 0, proc.stderr
    return out.read_text(), " ".join((proc.stdout + proc.stderr).split())


@pytest.mark.parametrize("archive", [None, ""], ids=["unset", "empty"])
def test_without_a_local_copy_the_release_url_is_used(tmp_path, archive):
    url, out = _url(tmp_path, archive)
    assert url == RELEASE_URL
    assert "BNGSIM_SUNDIALS_ARCHIVE" not in out


def test_a_named_file_is_used(tmp_path):
    archive = tmp_path / "sundials-7.2.1.tar.gz"
    archive.write_bytes(b"not checked here")
    url, out = _url(tmp_path, str(archive))
    assert url == archive.as_posix()
    assert "BNGSIM_SUNDIALS_ARCHIVE" not in out


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_a_name_that_is_not_a_file_falls_back_with_a_warning(tmp_path, kind):
    archive = tmp_path / "sundials-7.2.1.tar.gz"
    if kind == "directory":
        archive.mkdir()
    url, out = _url(tmp_path, str(archive))
    assert url == RELEASE_URL
    assert "which is not a file. Downloading" in out


def test_a_relative_name_falls_back_with_a_warning(tmp_path, monkeypatch):
    """Relative to what? The backend's configure directory is not the user's."""
    (tmp_path / "sundials-7.2.1.tar.gz").write_bytes(b"not checked here")
    monkeypatch.chdir(tmp_path)
    url, out = _url(tmp_path, "sundials-7.2.1.tar.gz")
    assert url == RELEASE_URL
    assert "which is not an absolute path" in out


def _archive(path: Path) -> str:
    """A small source archive shaped like SUNDIALS's: one top directory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = b"/* cvodes.c */\n"
        info = tarfile.TarInfo("sundials-0/src/cvodes/cvodes.c")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    path.write_bytes(buf.getvalue())
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _configure(tmp_path: Path, archive: Path, sha256: str) -> subprocess.CompletedProcess:
    """Configure a project that declares SUNDIALS the way CMakeLists.txt does,
    with a release URL no network can reach."""
    src = tmp_path / "project"
    src.mkdir(exist_ok=True)
    (src / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(fetch_check LANGUAGES NONE)\n"
        "include(FetchContent)\n"
        f'include("{MODULE.as_posix()}")\n'
        f'bngsim_sundials_source_url(_url "{RELEASE_URL}")\n'
        "FetchContent_Declare(sundials\n"
        '    URL "${_url}"\n'
        f'    URL_HASH "SHA256={sha256}"\n'
        "    DOWNLOAD_EXTRACT_TIMESTAMP FALSE\n"
        '    PATCH_COMMAND "${CMAKE_COMMAND}" -E touch <SOURCE_DIR>/patched)\n'
        "FetchContent_MakeAvailable(sundials)\n"
        'file(WRITE "${CMAKE_BINARY_DIR}/source_dir.txt" "${sundials_SOURCE_DIR}")\n'
    )
    return subprocess.run(
        [CMAKE, "-S", str(src), "-B", str(tmp_path / "build")],
        capture_output=True,
        text=True,
        timeout=300,
        env=_env(str(archive)),
    )


def test_fetchcontent_unpacks_and_patches_the_local_copy_offline(tmp_path):
    archive = tmp_path / "sundials-0.tar.gz"
    sha256 = _archive(archive)
    proc = _configure(tmp_path, archive, sha256)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    source = Path((tmp_path / "build" / "source_dir.txt").read_text())
    assert (source / "src" / "cvodes" / "cvodes.c").read_bytes() == b"/* cvodes.c */\n"
    assert (source / "patched").is_file()


def test_a_local_copy_with_other_bytes_stops_the_configure(tmp_path):
    archive = tmp_path / "sundials-0.tar.gz"
    sha256 = _archive(archive)
    archive.write_bytes(archive.read_bytes() + b"\0")
    proc = _configure(tmp_path, archive, sha256)
    assert proc.returncode != 0
    out = (proc.stdout + proc.stderr).lower()
    assert "hash" in out and sha256 in out


def test_cmake_declares_sundials_from_the_modules_answer():
    text = CMAKE_LISTS.read_text()
    call = text.index("bngsim_sundials_source_url(")
    args = text[call : text.index(")", call)].split("(", 1)[1].split()
    assert args == ["_bngsim_sundials_url", '"${BNGSIM_SUNDIALS_SOURCE_URL}"']
    assert 'include("${BNGSIM_SUNDIALS_DIR}/source_archive.cmake")' in text[:call]
    declare = text.index("FetchContent_Declare(\n        sundials")
    assert call < declare
    block = text[declare : text.index("FetchContent_MakeAvailable(sundials)")]
    assert 'URL "${_bngsim_sundials_url}"' in block
    assert 'URL_HASH "SHA256=${BNGSIM_SUNDIALS_SOURCE_SHA256}"' in block
