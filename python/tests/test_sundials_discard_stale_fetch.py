"""A build tree discards a SUNDIALS fetch that predates the current carry (#1301).

FetchContent runs the carry's patch step when it fetches SUNDIALS, and again
only if the step's command line changes; the carry's marker revision is in the
script, not on that line. So when the marker moves (r2 to r3 in #1409), every
existing build tree would keep its older carry, and the marker check after
FetchContent_MakeAvailable would stop its configure.
third_party/sundials/discard_stale_fetch.cmake runs first and removes such a
fetch, so FetchContent fetches and patches the source afresh.

These tests run the real module under real cmake -P against fake build trees.
Each pins one rule:

* a fetched cvodes.c without the current marker is discarded together with
  FetchContent's record of having fetched it, in both places that record can
  live: <_deps>/sundials-subbuild (policy CMP0168 OLD, bngsim's default) and
  CMakeFiles/fc-stamp/sundials and fc-tmp/sundials (CMP0168 NEW). Left in
  place, either record makes FetchContent skip the fetch, and the configure
  fails on a missing cvodes.c every time after. Nothing else is removed,
  sundials-build included;
* a fetch that already carries the marker is left alone;
* a FETCHCONTENT_SOURCE_DIR_SUNDIALS copy is the developer's own, and a
  FETCHCONTENT_FULLY_DISCONNECTED build cannot fetch again, so neither
  discards anything (the marker check refuses both instead);
* CMakeLists.txt calls the module before FetchContent_MakeAvailable(sundials),
  with the marker its own check uses.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = REPO_ROOT / "third_party" / "sundials" / "discard_stale_fetch.cmake"
CMAKE_LISTS = REPO_ROOT / "CMakeLists.txt"
VENDOR_JSON = REPO_ROOT / "third_party" / "sundials" / "VENDOR.json"

CMAKE = shutil.which("cmake")

pytestmark = [
    pytest.mark.skipif(
        not MODULE.exists(),
        reason="discard_stale_fetch.cmake is not in this checkout (installed package)",
    ),
    pytest.mark.skipif(CMAKE is None, reason="no CMake on PATH to run the module with"),
]

# Paths relative to the build tree's binary dir.
DISCARDED = {
    "_deps/sundials-src",
    "_deps/sundials-subbuild",
    "CMakeFiles/fc-stamp/sundials",
    "CMakeFiles/fc-tmp/sundials",
}
KEPT = {
    "_deps/sundials-build",
    "_deps/sundials-tmp",
    "_deps/klu-src",
    "_deps/klu-subbuild",
    "CMakeFiles/fc-stamp/klu",
    "CMakeFiles/fc-tmp/klu",
}


def _marker() -> str:
    (carry,) = json.loads(VENDOR_JSON.read_text())["local_carries"]
    return carry["marker"]


def _tree(tmp_path: Path, cvodes_text: str) -> Path:
    binary = tmp_path / "build"
    for rel in DISCARDED | KEPT:
        (binary / rel).mkdir(parents=True)
        (binary / rel / "stamp").write_text(rel)
    cvodes = binary / "_deps" / "sundials-src" / "src" / "cvodes" / "cvodes.c"
    cvodes.parent.mkdir(parents=True)
    cvodes.write_text(cvodes_text)
    return binary


def _run(tmp_path: Path, binary: Path, *, own: str = "", disconnected: str = "OFF") -> str:
    driver = tmp_path / "driver.cmake"
    driver.write_text(
        f'include("{MODULE.as_posix()}")\n'
        'bngsim_discard_stale_sundials_fetch("${BASE}" "${BIN}" "${MARKER}" "${OWN}" "${DISC}")\n'
    )
    proc = subprocess.run(
        [
            CMAKE,
            f"-DBASE={(binary / '_deps').as_posix()}",
            f"-DBIN={binary.as_posix()}",
            f"-DMARKER={_marker()}",
            f"-DOWN={own}",
            f"-DDISC={disconnected}",
            "-P",
            str(driver),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout + proc.stderr


def _present(binary: Path) -> set[str]:
    return {rel for rel in DISCARDED | KEPT if (binary / rel).exists()}


@pytest.mark.parametrize(
    "text",
    [
        "/* BNGSIM LOCAL CARRY (lanl/bngsim#1301, r2): ... */",
        "/* upstream cvodes.c, fetched before the carry existed */",
    ],
    ids=["older-revision", "no-carry"],
)
def test_a_stale_fetch_is_discarded_with_its_record(tmp_path, text):
    binary = _tree(tmp_path, text)
    out = _run(tmp_path, binary)
    assert _present(binary) == KEPT
    assert "fetched and patched again" in out


def test_a_current_fetch_is_left_alone(tmp_path):
    binary = _tree(tmp_path, f"/* {_marker()}: ... */")
    out = _run(tmp_path, binary)
    assert _present(binary) == DISCARDED | KEPT
    assert "fetched and patched again" not in out


def test_no_fetch_yet_is_a_no_op(tmp_path):
    binary = tmp_path / "build"
    (binary / "_deps").mkdir(parents=True)
    _run(tmp_path, binary)
    assert [p.name for p in binary.iterdir()] == ["_deps"]
    assert list((binary / "_deps").iterdir()) == []


@pytest.mark.parametrize(
    "kwargs",
    [{"own": "/somewhere/sundials-7.2.1"}, {"disconnected": "ON"}],
    ids=["own-source-dir", "fully-disconnected"],
)
def test_a_build_that_cannot_refetch_keeps_its_source(tmp_path, kwargs):
    binary = _tree(tmp_path, "/* BNGSIM LOCAL CARRY (lanl/bngsim#1301, r2): ... */")
    _run(tmp_path, binary, **kwargs)
    assert _present(binary) == DISCARDED | KEPT


def test_cmake_discards_before_it_fetches():
    """The call site passes what the module expects, in its order: a wrong
    argument here would discard nothing (or the wrong tree) with no error."""
    text = CMAKE_LISTS.read_text()
    call = text.index("bngsim_discard_stale_sundials_fetch(")
    assert call < text.index("FetchContent_MakeAvailable(sundials)")
    args = text[call : text.index(")", call)].split("(", 1)[1].split()
    assert args == [
        '"${FETCHCONTENT_BASE_DIR}"',
        '"${CMAKE_BINARY_DIR}"',
        '"${_bngsim_carry_marker}"',
        '"${FETCHCONTENT_SOURCE_DIR_SUNDIALS}"',
        '"${FETCHCONTENT_FULLY_DISCONNECTED}"',
    ]
    assert f'set(_bngsim_carry_marker "{_marker()}")' in text
    assert text.count(_marker()) == 1
