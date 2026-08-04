"""`scripts/ship_wheel.py`'s build-backend probe must track pyproject.

`_has_build_deps` decides whether an interpreter can run
``pip wheel --no-build-isolation``, and it answers by importing a hardcoded list
of module names. It has to be a hardcoded list — the probe runs against a
*foreign* interpreter that may have nothing installed, so it cannot ask that
interpreter's metadata what the build requires.

That makes `BUILD_DEP_MODULES` a second copy of ``[build-system] requires``, and
a wrong copy fails in the direction that hurts: name a module the build no
longer needs and every interpreter looks unable to build unisolated (slow but
correct); miss one the build *does* need and the probe waves through an
interpreter that then dies inside pip, which is the bug this file exists to
prevent recurring.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib", reason="tomllib is 3.11+")

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
SHIP_WHEEL = REPO_ROOT / "scripts" / "ship_wheel.py"


def _load_ship_wheel():
    """Import ship_wheel.py by path — scripts/ is not a package.

    The module must be in ``sys.modules`` *before* it executes: it defines a
    ``@dataclass``, and dataclasses resolves field types through
    ``sys.modules[cls.__module__]``, which is ``None`` for an unregistered module.
    """
    name = "_ship_wheel_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SHIP_WHEEL)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules[name]
        raise
    return mod


def _pyproject_build_requires() -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return data["build-system"]["requires"]


def _dist_name(requirement: str) -> str:
    """`scikit-build-core>=0.10` -> `scikit-build-core`."""
    return re.split(r"[<>=!~;\[\s]", requirement, maxsplit=1)[0].strip()


@pytest.mark.skipif(not SHIP_WHEEL.exists(), reason="ship_wheel.py not in this checkout")
def test_build_dep_modules_match_pyproject_requires():
    dists = {_dist_name(r) for r in _pyproject_build_requires()}
    expected_modules = {d.replace("-", "_") for d in dists}

    mod = _load_ship_wheel()

    assert set(mod.BUILD_DEP_MODULES) == expected_modules, (
        "scripts/ship_wheel.py BUILD_DEP_MODULES has drifted from pyproject's "
        f"[build-system] requires.\n  pyproject: {sorted(expected_modules)}\n"
        f"  ship_wheel: {sorted(mod.BUILD_DEP_MODULES)}"
    )


@pytest.mark.skipif(not SHIP_WHEEL.exists(), reason="ship_wheel.py not in this checkout")
def test_build_dep_dists_names_the_same_packages():
    """The human-readable string in the error message must not drift either."""
    mod = _load_ship_wheel()
    named = {s.strip() for s in mod.BUILD_DEP_DISTS.split(",")}
    assert named == {_dist_name(r) for r in _pyproject_build_requires()}


@pytest.fixture
def pip_but_no_backend(tmp_path):
    """A stub interpreter that has pip and nothing else.

    Stands in for a bare ``uv python install`` interpreter. A stub rather than a
    real venv so the test costs milliseconds and does not depend on what happens
    to be installed on the machine.
    """
    stub = tmp_path / "fake_python"
    stub.write_text(
        "#!/bin/sh\n"
        # Succeed for `import pip`, fail for anything naming the build backend.
        'case "$*" in\n'
        "  *scikit_build_core*|*pybind11*) exit 1 ;;\n"
        "  *) exit 0 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub


def test_probe_rejects_an_interpreter_without_the_backend(pip_but_no_backend):
    """The two probes disagree on exactly the interpreter that caused the bug."""
    mod = _load_ship_wheel()

    assert mod._has_pip(str(pip_but_no_backend)) is True
    assert mod._has_build_deps(str(pip_but_no_backend)) is False


def test_build_command_isolates_when_backend_is_missing(monkeypatch, tmp_path, pip_but_no_backend):
    """pip present + backend absent must NOT select ``--no-build-isolation``.

    Driven through the real probes against a stub interpreter, deliberately: an
    equivalent test that monkeypatches `_has_build_deps` would fail on the
    pre-fix script only because that helper did not exist yet, which pins the
    refactor rather than the behaviour. This one fails on the pre-fix script by
    selecting the wrong command, which is the defect.
    """
    mod = _load_ship_wheel()
    monkeypatch.setattr(mod.sys, "executable", str(pip_but_no_backend))
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    cmd = mod._build_command(tmp_path)

    assert "--no-build-isolation" not in cmd
    assert cmd[:2] == ["uv", "build"]


def test_build_command_uses_pip_when_backend_is_present(monkeypatch, tmp_path):
    """The canonical path is unchanged for a properly provisioned dev venv."""
    mod = _load_ship_wheel()
    monkeypatch.setattr(mod, "_has_pip", lambda exe: True)
    monkeypatch.setattr(mod, "_has_build_deps", lambda exe: True)

    cmd = mod._build_command(tmp_path)

    assert "--no-build-isolation" in cmd
    assert cmd[1:3] == ["-m", "pip"]


def test_error_names_the_missing_build_deps(monkeypatch, tmp_path):
    """With no uv to fall back to, the message must say what is missing."""
    mod = _load_ship_wheel()
    monkeypatch.setattr(mod, "_has_pip", lambda exe: True)
    monkeypatch.setattr(mod, "_has_build_deps", lambda exe: False)
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match=r"scikit-build-core"):
        mod._build_command(tmp_path)
