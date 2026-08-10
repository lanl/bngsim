"""`scripts/rebuild_editable.py` must be able to find pybind11 (GH #229).

That script is the *fast* way to refresh the editable C++ extension, and — more
to the point — it is the remedy the stale-binary guard names
(`_build_provenance.format_report`). It drives cmake directly against the
environment it is running in, so `find_package(pybind11 CONFIG REQUIRED)` has to
resolve from that environment. But pybind11 is declared only in
`[build-system] requires`: uv supplies it in a transient isolated build env and
never installs it into `.venv`. So the script's configure had nothing pointing at
pybind11 and worked only where the machine happened to carry a system-wide copy,
or where the venv had kept one by accident of history until the next `uv sync`
pruned it.

The consequence is what makes this more than a broken helper. The only other
thing the guard offers is `BNGSIM_ALLOW_STALE_CORE=1` — proceed against a binary
that does not match the source, which is exactly what the guard exists to
prevent. A remedy that fails there does not send the reader looking for a better
one; it sends them to the escape hatch.

Two halves, tested here:

* the script asks its own interpreter and pins `-Dpybind11_DIR`, which both makes
  the configure work where nothing else supplies pybind11 *and* stops a
  cmake-only rebuild from silently compiling against an unrelated system copy;
* the `dev` extra declares pybind11, with the same specifier as
  `[build-system] requires` — one dependency reached by two routes, and a skew
  means the two documented rebuild paths can build the same source against
  different pybind11 versions.

The absent-pybind11 case is deliberately *not* a refusal: cmake can still find a
system pybind11, and plenty of machines rebuild that way today. Breaking them for
a dependency they demonstrably do not need would trade one failure for another.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib", reason="tomllib is 3.11+")

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
REBUILD_EDITABLE = REPO_ROOT / "scripts" / "rebuild_editable.py"

pytestmark = pytest.mark.skipif(
    not REBUILD_EDITABLE.exists(),
    reason="scripts/rebuild_editable.py is not in this checkout (installed package)",
)


def _load_rebuild_editable() -> types.ModuleType:
    """Import rebuild_editable.py by path — scripts/ is not a package."""
    name = "_rebuild_editable_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REBUILD_EDITABLE)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules[name]
        raise
    return mod


@pytest.fixture
def stub_pybind11(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """An importable pybind11 whose CMake dir is ours, in any environment.

    A stub rather than the real package on purpose: the environment that runs
    this suite is provisioned with `--extra test`, which does *not* carry
    pybind11, so a test gated on the real one would skip in CI and in every
    worktree — precisely where the regression would go unnoticed.
    """
    cmake_dir = tmp_path / "share" / "cmake" / "pybind11"
    cmake_dir.mkdir(parents=True)
    (cmake_dir / "pybind11Config.cmake").write_text("", encoding="utf-8")
    mod = types.ModuleType("pybind11")
    mod.get_cmake_dir = lambda: str(cmake_dir)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pybind11", mod)
    return cmake_dir


@pytest.fixture
def no_pybind11(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `import pybind11` raise, whatever the environment actually has.

    ``None`` in ``sys.modules`` is the documented way to poison an import: the
    import system raises ImportError instead of consulting the finders.
    """
    monkeypatch.setitem(sys.modules, "pybind11", None)  # type: ignore[misc]


@pytest.fixture
def captured_cmake(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[list[str]]:
    """Run `main()` for real, with every subprocess recorded instead of run.

    Driving `main()` rather than the argv helper directly is deliberate: the
    defect is *what the configure line contains*, and a test that called the
    helper would pass trivially on a version of the script where main() never
    consults it.
    """
    mod = _load_rebuild_editable()
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> None:
        calls.append(list(cmd))

    monkeypatch.setattr(mod, "_run", fake_run)
    monkeypatch.setattr(mod, "_load_build_info", lambda src: {"build_dir": str(tmp_path / "bld")})
    monkeypatch.setattr(mod, "_install_prefix", lambda: tmp_path / "prefix")
    monkeypatch.setattr(mod, "_regenerate_stub", lambda *a, **k: None)
    monkeypatch.setenv("BNGSIM_SKIP_METADATA_REFRESH", "1")
    return calls


def _configure_line(calls: list[list[str]]) -> list[str]:
    configure = [c for c in calls if c and c[0] == "cmake" and "-S" in c]
    assert len(configure) == 1, f"expected exactly one configure, got {calls}"
    return configure[0]


# ── The configure line ────────────────────────────────────────────────────────


def test_configure_pins_the_running_interpreters_pybind11(
    stub_pybind11: Path, captured_cmake: list[list[str]]
) -> None:
    mod = _load_rebuild_editable()

    assert mod.main() == 0

    assert f"-Dpybind11_DIR={stub_pybind11}" in _configure_line(captured_cmake)


def test_configure_still_pins_the_interpreter(
    stub_pybind11: Path, captured_cmake: list[list[str]]
) -> None:
    """The pybind11 pin is an addition; the pins that were already load-bearing stay."""
    mod = _load_rebuild_editable()

    assert mod.main() == 0

    configure = _configure_line(captured_cmake)
    assert f"-DPython_EXECUTABLE={sys.executable}" in configure
    assert f"-DPython3_EXECUTABLE={sys.executable}" in configure
    assert "-DBNGSIM_BUILD_PYTHON=ON" in configure


def test_configure_leaves_discovery_to_cmake_when_pybind11_is_absent(
    no_pybind11: None, captured_cmake: list[list[str]], capsys: pytest.CaptureFixture[str]
) -> None:
    """No pybind11 must not become a refusal — a system copy still works."""
    mod = _load_rebuild_editable()

    assert mod.main() == 0

    configure = _configure_line(captured_cmake)
    assert not any(arg.startswith("-Dpybind11_DIR=") for arg in configure)
    # ...but it is said out loud, before the configure runs, so a build against
    # some unpinned system pybind11 is visible rather than silent.
    assert "pybind11: not importable" in capsys.readouterr().out


def test_configure_failure_without_pybind11_names_the_remedy(
    no_pybind11: None,
    captured_cmake: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raw CMake error 900 lines into someone else's build system is not a remedy."""
    import subprocess

    mod = _load_rebuild_editable()

    def fail_configure(cmd: list[str], **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(mod, "_run", fail_configure)

    with pytest.raises(SystemExit) as excinfo:
        mod.main()

    message = str(excinfo.value)
    assert "pybind11" in message
    assert "uv sync --extra dev" in message


def test_configure_failure_with_pybind11_is_left_alone(
    stub_pybind11: Path,
    captured_cmake: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With pybind11 pinned, a failing configure is some other bug — do not misdiagnose it."""
    import subprocess

    mod = _load_rebuild_editable()

    def fail_configure(cmd: list[str], **kwargs: object) -> None:
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(mod, "_run", fail_configure)

    with pytest.raises(subprocess.CalledProcessError):
        mod.main()


# ── Resolving pybind11 ────────────────────────────────────────────────────────


def test_cmake_dir_is_none_when_pybind11_cannot_be_imported(no_pybind11: None) -> None:
    mod = _load_rebuild_editable()
    assert mod._pybind11_cmake_dir() is None


def test_cmake_dir_is_none_when_the_directory_holds_no_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pinning a config-less directory swaps one cmake error for a stranger one."""
    stub = types.ModuleType("pybind11")
    stub.get_cmake_dir = lambda: str(tmp_path)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pybind11", stub)

    mod = _load_rebuild_editable()
    assert mod._pybind11_cmake_dir() is None


@pytest.mark.skipif(
    importlib.util.find_spec("pybind11") is None,
    reason="pybind11 is not installed (it ships in the dev extra)",
)
def test_real_pybind11_resolves_to_a_usable_cmake_dir() -> None:
    """The stub's contract, checked once against the actual package."""
    mod = _load_rebuild_editable()
    cmake_dir = mod._pybind11_cmake_dir()
    assert cmake_dir is not None
    assert (Path(cmake_dir) / "pybind11Config.cmake").is_file()


# ── Per-build stamps in the committed stub ────────────────────────────────────


def test_stub_normalization_covers_every_per_build_stamp() -> None:
    """pybind11-stubgen copies build facts into a committed file; both get pinned.

    ``__pybind11_version__`` (GH #288) joined ``__build_commit__`` as a value that
    differs per build environment, so it is the same spurious-diff hazard PR #70
    merged once already — and here the differing value would be *the* symptom
    #288 exists to surface, flip-flopping in the stub instead of being read off
    the binary.
    """
    mod = _load_rebuild_editable()
    generated = "\n".join(
        [
            "HAS_KLU: bool = True",
            "__build_commit__: str = 'e61f83d57358+dirty'",
            "__pybind11_version__: str = '3.0.4'",
            "__version__: str = '0.12.2'",
        ]
    )

    normalized = mod._normalize_stub_build_stamps(generated)

    assert "__build_commit__: str = 'unknown'" in normalized
    assert "__pybind11_version__: str = 'unknown'" in normalized
    # Only the per-build stamps: the package version is an API fact, not a build one.
    assert "__version__: str = '0.12.2'" in normalized
    assert "HAS_KLU: bool = True" in normalized


def test_stub_normalization_is_a_noop_when_a_stamp_is_absent() -> None:
    mod = _load_rebuild_editable()
    generated = "__version__: str = '0.12.2'\n"

    assert mod._normalize_stub_build_stamps(generated) == generated


# ── The remedy text and the extra behind it ───────────────────────────────────


def test_missing_note_names_a_command_that_provisions_pybind11() -> None:
    mod = _load_rebuild_editable()
    note = mod._PYBIND11_MISSING_NOTE

    assert "uv sync --extra dev" in note
    # The isolated-build path too: it needs no local pybind11 at all, and it is
    # what CONTRIBUTING.md documents.
    assert "--reinstall-package bngsim" in note
    # Naming the extras on that line is the point — a bare
    # `uv sync --reinstall-package bngsim` prunes them, which is how the
    # reported breakage started.
    assert re.search(r"uv sync --extra \S+ --reinstall-package bngsim", note)


def _requirements(section: list[str], dist: str) -> list[str]:
    return [r for r in section if re.split(r"[<>=!~;\[\s]", r, maxsplit=1)[0].strip() == dist]


def test_dev_extra_declares_pybind11_with_the_build_system_specifier() -> None:
    """One dependency, two routes into a build; a skew builds two different things."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    build_requires = _requirements(data["build-system"]["requires"], "pybind11")
    dev = _requirements(data["project"]["optional-dependencies"]["dev"], "pybind11")

    assert len(build_requires) == 1, "pyproject's [build-system] requires must name pybind11 once"
    assert dev == build_requires, (
        "the `dev` extra must declare pybind11 with the same specifier as "
        f"[build-system] requires — got dev={dev}, build-system={build_requires}"
    )
