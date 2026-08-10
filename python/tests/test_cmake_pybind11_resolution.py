"""Which pybind11 compiles the extension is decided here (GH #288).

`cmake/BngsimResolvePybind11.cmake` picks the interpreter that answers
`find_package(pybind11)`, and every wrong answer it can give is silent: any
recent pybind11 produces a module that builds, installs and imports. Until #288
the candidate list did not include `Python_EXECUTABLE` — the interpreter
scikit-build-core is actually building for — and reached
`${CMAKE_SOURCE_DIR}/.venv/bin/python` third. On any machine whose checkout has a
`.venv`, that entry won every build that did not set `BNGSIM_PYTHON_EXECUTABLE`
or `VIRTUAL_ENV`, including isolated wheel builds whose own `[build-system]
requires` had already resolved and installed a *different* pybind11. Two wheels
built from one commit, one of them for a completely different interpreter, came
out compiled against the dev venv's 3.0.4 either way.

So these tests are about ordering, and they run the real module under real cmake
against fake interpreters — a shell script that prints a directory is all the
probe (`python -c "import pybind11; print(pybind11.get_cmake_dir())"`) can tell
apart from the real thing. That is what makes the ordering testable at all: the
alternative is configuring the whole project once per case against interpreters
that would have to genuinely carry different pybind11 versions.

Each case pins one rule:

* the interpreter the build targets wins (the #288 regression);
* ...but only while it can actually answer, because the fallback list is what
  keeps `scripts/rebuild_editable.py` working after the build-isolation venv it
  was configured in has been deleted (#23, #229) — the reason the fix is a
  prepend and not a replacement;
* a *deleted* target interpreter is dropped before it can shadow that fallback;
* an explicit pin (`-Dpybind11_DIR`, `$BNGSIM_PYTHON_EXECUTABLE`) still wins
  outright, since both exist precisely to overrule this logic.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = REPO_ROOT / "cmake" / "BngsimResolvePybind11.cmake"

CMAKE = shutil.which("cmake")

pytestmark = [
    pytest.mark.skipif(
        not MODULE.exists(),
        reason="cmake/BngsimResolvePybind11.cmake is not in this checkout (installed package)",
    ),
    pytest.mark.skipif(CMAKE is None, reason="no CMake on PATH to configure the probe with"),
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="the fake interpreters are /bin/sh scripts, so this probe is POSIX-specific",
    ),
]

_PROJECT = """\
cmake_minimum_required(VERSION 3.20)
project(bngsim_pybind11_probe NONE)
include("${BNGSIM_MODULE}")
bngsim_resolve_pybind11_dir()
file(WRITE "${CMAKE_BINARY_DIR}/resolved.txt" "${pybind11_DIR}")
"""


# ── Fixtures for a configure ──────────────────────────────────────────────────


class Probe:
    """A scratch CMake project plus the fake interpreters it may consult."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.source = root / "src"
        self.source.mkdir(parents=True)
        (self.source / "CMakeLists.txt").write_text(_PROJECT, encoding="utf-8")
        self.log = root / "probed.log"
        self.last_build = root
        self.last_output = ""
        self._runs = 0

    # -- things a candidate can be --------------------------------------------

    def pybind11_dir(self, name: str, *, with_config: bool = True) -> Path:
        """A directory shaped like pybind11's installed CMake package."""
        d = self.root / name / "share" / "cmake" / "pybind11"
        d.mkdir(parents=True)
        if with_config:
            (d / "pybind11Config.cmake").write_text("", encoding="utf-8")
        return d

    def interpreter(self, rel: str, answer: Path | None) -> Path:
        """A fake python at ``rel`` that prints ``answer`` (or fails like no pybind11).

        Every invocation appends its own path to the probe log, so a test can
        assert not just the winner but that the walk stopped there.
        """
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        body = f'echo "{answer}"\n' if answer is not None else "exit 1\n"
        path.write_text(
            f'#!/bin/sh\necho "{path}" >> "{self.log}"\n{body}',
            encoding="utf-8",
        )
        path.chmod(0o755)
        return path

    # -- running it -----------------------------------------------------------

    def configure(
        self,
        *defines: str,
        env: dict[str, str] | None = None,
        search_env_path: bool = False,
    ) -> str:
        """Configure once; return the resolved ``pybind11_DIR`` (``""`` if none).

        The two ``CMAKE_FIND_USE_*`` knobs make the last-resort ``find_program``
        candidate hermetic: without them it would pick up whatever ``python3``
        the host happens to have, and whether that one carries pybind11 would
        decide the outcome of the negative cases.

        They also stop CMake finding ``make``, which it insists on even for a
        ``project(NONE)`` that will never be built — so the build program is
        named outright rather than searched for. ``/bin/sh`` is never invoked;
        it is just a path that exists. Naming it is what lets this run on a host
        with no make and no ninja at all, which the macOS CI runner is: cmake
        arrives there as a pip wheel, and nothing else does.
        """
        self._runs += 1
        build = self.root / f"build{self._runs}"
        full_env = dict(os.environ)
        full_env.pop("BNGSIM_PYTHON_EXECUTABLE", None)
        full_env.pop("VIRTUAL_ENV", None)
        full_env.update(env or {})
        assert CMAKE is not None
        proc = subprocess.run(
            [
                CMAKE,
                "-S",
                str(self.source),
                "-B",
                str(build),
                f"-DBNGSIM_MODULE={MODULE}",
                "-DCMAKE_MAKE_PROGRAM=/bin/sh",
                "-DCMAKE_FIND_USE_SYSTEM_ENVIRONMENT_PATH=" + ("ON" if search_env_path else "OFF"),
                "-DCMAKE_FIND_USE_CMAKE_SYSTEM_PATH=OFF",
                *defines,
            ],
            env=full_env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"configure failed:\n{proc.stdout}\n{proc.stderr}"
        self.last_build = build
        self.last_output = proc.stdout
        return (build / "resolved.txt").read_text(encoding="utf-8").strip()

    def probed(self) -> list[str]:
        """Interpreters the configure actually executed, in order."""
        if not self.log.exists():
            return []
        return self.log.read_text(encoding="utf-8").splitlines()

    def cache(self) -> str:
        return (self.last_build / "CMakeCache.txt").read_text(encoding="utf-8")


@pytest.fixture
def probe(tmp_path: Path) -> Probe:
    return Probe(tmp_path)


# ── The #288 ordering ─────────────────────────────────────────────────────────


def test_target_interpreter_beats_a_venv_in_the_checkout(probe: Probe) -> None:
    """The whole defect: a wheel build for interpreter A compiled against B's pybind11."""
    target = probe.pybind11_dir("target")
    checkout = probe.pybind11_dir("checkout")
    py = probe.interpreter("target/bin/python", target)
    probe.interpreter("src/.venv/bin/python", checkout)

    resolved = probe.configure(f"-DPython_EXECUTABLE={py}")

    assert resolved == str(target)
    # ...and the checkout venv was never even asked.
    assert probe.probed() == [str(py)]


@pytest.mark.parametrize("spelling", ["Python_EXECUTABLE", "Python3_EXECUTABLE"])
def test_every_spelling_of_the_target_interpreter_is_consulted(
    probe: Probe, spelling: str
) -> None:
    """FindPython caches all three names; scikit-build-core sets the first two."""
    target = probe.pybind11_dir("target")
    checkout = probe.pybind11_dir("checkout")
    py = probe.interpreter("target/bin/python", target)
    probe.interpreter("src/.venv/bin/python", checkout)

    assert probe.configure(f"-D{spelling}={py}") == str(target)


def test_checkout_venv_answers_when_the_target_interpreter_cannot(probe: Probe) -> None:
    """The fallback list is load-bearing: pybind11 is a build-system requirement only.

    `uv` never installs it into `.venv`, so the interpreter a plain cmake rebuild
    targets routinely has no pybind11 in it. That must fall through, not fail.
    """
    checkout = probe.pybind11_dir("checkout")
    py = probe.interpreter("target/bin/python", None)
    venv_py = probe.interpreter("src/.venv/bin/python", checkout)

    resolved = probe.configure(f"-DPython_EXECUTABLE={py}")

    assert resolved == str(checkout)
    assert probe.probed() == [str(py), str(venv_py)]


def test_deleted_target_interpreter_is_dropped_not_preferred(probe: Probe) -> None:
    """A uv build-isolation venv is gone by the time rebuild_editable.py re-runs cmake.

    Its `Python_EXECUTABLE` is still in the cache and now names nothing. Promoting
    that path to first place without dropping it would have made this the phantom
    the #23 guard was written for.
    """
    checkout = probe.pybind11_dir("checkout")
    venv_py = probe.interpreter("src/.venv/bin/python", checkout)
    phantom = probe.root / "builds-v0" / ".tmpXXXX" / "bin" / "python"

    resolved = probe.configure(f"-DPython_EXECUTABLE={phantom}")

    assert resolved == str(checkout)
    assert probe.probed() == [str(venv_py)]
    assert str(phantom) not in probe.cache(), "the dead path survived in the cache"


def test_env_override_outranks_the_target_interpreter(probe: Probe) -> None:
    """$BNGSIM_PYTHON_EXECUTABLE is the escape hatch for a machine that resolves wrong.

    It only works as one by sitting ahead of everything this module would
    otherwise decide for itself.
    """
    override = probe.pybind11_dir("override")
    target = probe.pybind11_dir("target")
    override_py = probe.interpreter("override/bin/python", override)
    py = probe.interpreter("target/bin/python", target)

    resolved = probe.configure(
        f"-DPython_EXECUTABLE={py}",
        env={"BNGSIM_PYTHON_EXECUTABLE": str(override_py)},
    )

    assert resolved == str(override)


def test_virtualenv_is_tried_before_the_checkout(probe: Probe) -> None:
    active = probe.pybind11_dir("active")
    checkout = probe.pybind11_dir("checkout")
    active_py = probe.interpreter("active/bin/python", active)
    probe.interpreter("src/.venv/bin/python", checkout)

    resolved = probe.configure(env={"VIRTUAL_ENV": str(active_py.parent.parent)})

    assert resolved == str(active)


def test_path_python_is_the_last_resort(probe: Probe) -> None:
    """Nothing named an interpreter, so the one on PATH gets the question."""
    on_path = probe.pybind11_dir("onpath")
    py = probe.interpreter("bin/python3", on_path)

    resolved = probe.configure(env={"PATH": str(py.parent)}, search_env_path=True)

    assert resolved == str(on_path)


def test_windows_venv_layout_is_a_candidate(probe: Probe) -> None:
    """venvs put the interpreter in Scripts/ on Windows; only bin/ was listed."""
    checkout = probe.pybind11_dir("checkout")
    probe.interpreter("src/.venv/Scripts/python.exe", checkout)

    assert probe.configure() == str(checkout)


# ── Explicit pins and unusable answers ────────────────────────────────────────


def test_explicit_pybind11_dir_is_left_alone(probe: Probe) -> None:
    """`-Dpybind11_DIR` is how rebuild_editable.py pins its own answer (#229)."""
    pinned = probe.pybind11_dir("pinned")
    target = probe.pybind11_dir("target")
    py = probe.interpreter("target/bin/python", target)

    resolved = probe.configure(f"-Dpybind11_DIR={pinned}", f"-DPython_EXECUTABLE={py}")

    assert resolved == str(pinned)
    assert probe.probed() == [], "an explicit pin must short-circuit the walk entirely"


def test_pybind11_dir_pointing_at_nothing_is_re_resolved(probe: Probe) -> None:
    """The other half of the phantom cache: the pinned dir went away with its venv."""
    target = probe.pybind11_dir("target")
    py = probe.interpreter("target/bin/python", target)
    stale = probe.root / "gone" / "share" / "cmake" / "pybind11"

    resolved = probe.configure(f"-Dpybind11_DIR={stale}", f"-DPython_EXECUTABLE={py}")

    assert resolved == str(target)


def test_interpreter_answering_with_a_config_less_dir_is_skipped(probe: Probe) -> None:
    """Pinning a directory with no config file trades one cmake error for a stranger one.

    `find_package` would then complain about a bad `pybind11_DIR` rather than
    about pybind11 being missing — and the walk would have stopped at an
    interpreter that cannot actually supply it.
    """
    empty = probe.pybind11_dir("empty", with_config=False)
    checkout = probe.pybind11_dir("checkout")
    py = probe.interpreter("target/bin/python", empty)
    venv_py = probe.interpreter("src/.venv/bin/python", checkout)

    resolved = probe.configure(f"-DPython_EXECUTABLE={py}")

    assert resolved == str(checkout)
    assert probe.probed() == [str(py), str(venv_py)]


def test_no_candidate_answers_is_not_fatal(probe: Probe) -> None:
    """A system-wide pybind11 is a legitimate build; find_package still gets its turn."""
    py = probe.interpreter("target/bin/python", None)

    assert probe.configure(f"-DPython_EXECUTABLE={py}") == ""
    assert "leaving discovery to find_package" in probe.last_output


def test_the_resolving_interpreter_is_named_in_the_configure_log(probe: Probe) -> None:
    """#288 was invisible in every build log it ever happened in."""
    target = probe.pybind11_dir("target")
    py = probe.interpreter("target/bin/python", target)

    probe.configure(f"-DPython_EXECUTABLE={py}")

    assert f"pybind11 resolved via {py}" in probe.last_output
