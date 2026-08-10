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

Since GH #275 a failed probe costs speed rather than the build — pip's own
isolation supplies the backend — so the tests here also pin the two facts that
fallback rests on: which combination is genuinely unrecoverable, and that the
local wheel matrix builds isolated too, so an isolated fallback is not a
deviation from what the matrix validates.
"""

from __future__ import annotations

import importlib.util
import inspect
import re
import sys
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib", reason="tomllib is 3.11+")

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
SHIP_WHEEL = REPO_ROOT / "scripts" / "ship_wheel.py"
LOCAL_CI = REPO_ROOT / "scripts" / "local_ci.py"


def _load_script(name: str, path: Path):
    """Import a scripts/ module by path — scripts/ is not a package.

    The module must be in ``sys.modules`` *before* it executes: ship_wheel
    defines a ``@dataclass``, and dataclasses resolves field types through
    ``sys.modules[cls.__module__]``, which is ``None`` for an unregistered module.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules[name]
        raise
    return mod


def _load_ship_wheel():
    return _load_script("_ship_wheel_under_test", SHIP_WHEEL)


def _load_local_ci():
    return _load_script("_local_ci_under_test", LOCAL_CI)


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


def test_pip_without_the_backend_and_no_uv_builds_isolated(monkeypatch, tmp_path, capsys):
    """pip alone is enough: pip's own isolation supplies the backend (GH #275).

    This used to raise. It rested on `_build_command`'s claim that the
    unisolated form is "what the wheel matrix validates", which
    `test_local_ci_matrix_builds_isolated` shows is false — so refusing here
    bought no artifact fidelity and cost the build outright.
    """
    mod = _load_ship_wheel()
    monkeypatch.setattr(mod, "_has_pip", lambda exe: True)
    monkeypatch.setattr(mod, "_has_build_deps", lambda exe: False)
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)

    cmd = mod._build_command(tmp_path)

    assert cmd[1:4] == ["-m", "pip", "wheel"]
    assert "--no-build-isolation" not in cmd
    # The slow path is announced, and says how to get off it.
    assert "scikit-build-core" in capsys.readouterr().out


def test_no_pip_and_no_uv_still_raises(monkeypatch, tmp_path):
    """The one unrecoverable combination, and the message must name the fix."""
    mod = _load_ship_wheel()
    monkeypatch.setattr(mod, "_has_pip", lambda exe: False)
    monkeypatch.setattr(mod, "_has_build_deps", lambda exe: False)
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match=r"no pip.*uv is not on PATH"):
        mod._build_command(tmp_path)


@pytest.mark.skipif(not LOCAL_CI.exists(), reason="local_ci.py not in this checkout")
def test_local_ci_matrix_builds_isolated():
    """The claim `_build_command` now rests on: the wheel matrix builds isolated.

    `_build_command` falls back to an isolated build instead of refusing, and
    justifies it by saying isolation is what `scripts/LOCAL_CI.md` actually
    measures. That is a fact about `local_ci.py`, so pin it here: the matrix
    provisions its throwaway build venv with `build`/`cmake`/`ninja` and no PEP
    517 backend, then runs pypa/build with its default isolation. Adding
    `--no-isolation` there would silently invalidate the docstring.
    """
    mod = _load_local_ci()
    source = inspect.getsource(mod.build_wheel)

    assert '"build"' in source
    for opt_out in ("--no-isolation", "--no-build-isolation"):
        assert opt_out not in source, f"local_ci.py's matrix build now passes {opt_out}"
    for backend_dist in ("scikit-build-core", "pybind11"):
        assert backend_dist not in source, (
            f"local_ci.py's build venv now installs {backend_dist}; the matrix may no "
            "longer be building isolated, which ship_wheel._build_command relies on."
        )
