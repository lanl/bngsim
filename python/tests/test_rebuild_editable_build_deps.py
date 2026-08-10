"""`--no-build-isolation` needs the backend, not pip (GH #271).

`scripts/rebuild_editable.py:_editable_install_cmd` chose
``pip install --no-build-isolation`` whenever ``pip`` was importable. That is not
the precondition the flag has: an unisolated PEP 517 build needs pip **and**
``[build-system] requires`` importable in the same interpreter, and the two come
apart. A ``uv venv --seed`` interpreter has pip and no ``scikit_build_core``,
because the backend lives only in ``[build-system] requires`` and uv puts that in
a transient build env — the same fact behind GH #229. Measured on exactly such an
interpreter, the old command died with::

    BackendUnavailable: Cannot import 'scikit_build_core.build'

naming neither the missing dist nor the fix, from inside pip's vendored
``pyproject_hooks``.

`scripts/ship_wheel.py:_has_build_deps` was written to reject this same
inference ("Having pip is not it") — so this was one dependency question answered
in two files and fixed in one. The two must keep agreeing, which is what
`test_build_dep_modules_match_ship_wheels_hardcoded_copy` is for; ship_wheel has
to hardcode its copy because it probes *foreign* interpreters, while this script
only ever targets the one it runs in and reads the requirement out of pyproject.

The fix is not a refusal. Dropping the flag lets pip supply the backend itself,
which is a path that works — verified by running it — so no environment that
built before stops building. Only "no pip and no uv" is unrecoverable, and that
one already raised.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REBUILD_EDITABLE = REPO_ROOT / "scripts" / "rebuild_editable.py"
SHIP_WHEEL = REPO_ROOT / "scripts" / "ship_wheel.py"

pytestmark = pytest.mark.skipif(
    not REBUILD_EDITABLE.exists(),
    reason="scripts/rebuild_editable.py is not in this checkout (installed package)",
)


def _load(path: Path, name: str) -> types.ModuleType:
    """Import a scripts/ module by path — scripts/ is not a package."""
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


def _rebuild_editable() -> types.ModuleType:
    return _load(REBUILD_EDITABLE, "_rebuild_editable_build_deps_under_test")


def _present(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Make each name importable, with a real ``__spec__``.

    ``find_spec`` consults ``sys.modules`` first and raises on an entry whose
    ``__spec__`` is missing or None, so a bare ``ModuleType`` would read as
    *broken* rather than present and the test would pass for the wrong reason.
    """
    for name in names:
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
        monkeypatch.setitem(sys.modules, name, mod)


def _absent(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    """Make each name unimportable whatever the environment really has."""
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)  # type: ignore[misc]


# ── The list, and the copy of it that already existed ─────────────────────────


def test_build_dep_modules_match_pyprojects_build_system() -> None:
    """The hand-rolled parse must agree with a real TOML reader."""
    tomllib = pytest.importorskip("tomllib", reason="tomllib is 3.11+")
    import re as _re

    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    expected = tuple(
        _re.split(r"[<>=!~;\[\s]", r, maxsplit=1)[0].strip().replace("-", "_")
        for r in data["build-system"]["requires"]
    )

    assert _rebuild_editable()._build_dep_modules(REPO_ROOT) == expected


@pytest.mark.skipif(not SHIP_WHEEL.exists(), reason="ship_wheel.py not in this checkout")
def test_build_dep_modules_match_ship_wheels_hardcoded_copy() -> None:
    """Two scripts, one question. ship_wheel must hardcode it; this one need not."""
    ship_wheel = _load(SHIP_WHEEL, "_ship_wheel_for_build_deps")
    parsed = _rebuild_editable()._build_dep_modules(REPO_ROOT)

    assert parsed is not None
    assert set(parsed) == set(ship_wheel.BUILD_DEP_MODULES), (
        "scripts/ship_wheel.py BUILD_DEP_MODULES and rebuild_editable's parse of "
        f"[build-system] requires disagree — {sorted(ship_wheel.BUILD_DEP_MODULES)} "
        f"vs {sorted(parsed)}"
    )


def test_unparseable_build_system_assumes_the_deps_are_absent(tmp_path: Path) -> None:
    """The safe direction: an isolated build costs time, a wrong guess costs a build."""
    mod = _rebuild_editable()
    assert mod._build_dep_modules(tmp_path) is None
    assert mod._has_build_deps(tmp_path) is False

    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    assert mod._build_dep_modules(tmp_path) is None
    assert mod._has_build_deps(tmp_path) is False


def test_can_import_reports_a_poisoned_entry_as_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """``find_spec`` raises on these rather than returning None; both mean unusable."""
    mod = _rebuild_editable()
    _absent(monkeypatch, "pip")
    assert mod._can_import("pip") is False

    monkeypatch.setitem(sys.modules, "pip", types.ModuleType("pip"))  # __spec__ is None
    assert mod._can_import("pip") is False


# ── Which command comes out ───────────────────────────────────────────────────


def test_pip_with_the_backend_keeps_the_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The canonical path is untouched: unisolated reuses the configured build tree."""
    mod = _rebuild_editable()
    _present(monkeypatch, "pip", "scikit_build_core", "pybind11")

    cmd = mod._editable_install_cmd(REPO_ROOT)

    assert cmd[1:3] == ["-m", "pip"]
    assert "--no-build-isolation" in cmd


def test_pip_without_the_backend_drops_no_build_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The defect. pip is there, the backend is not, and the flag must come off."""
    mod = _rebuild_editable()
    _present(monkeypatch, "pip")
    _absent(monkeypatch, "scikit_build_core", "pybind11")

    cmd = mod._editable_install_cmd(REPO_ROOT)

    assert cmd[1:3] == ["-m", "pip"], "still pip — isolation is the only thing that changes"
    assert "--no-build-isolation" not in cmd
    assert "-e" in cmd


def test_one_missing_backend_module_is_enough(monkeypatch: pytest.MonkeyPatch) -> None:
    """`all()`, not `any()` — an unisolated build needs every requirement present."""
    mod = _rebuild_editable()
    _present(monkeypatch, "pip", "scikit_build_core")
    _absent(monkeypatch, "pybind11")

    assert "--no-build-isolation" not in mod._editable_install_cmd(REPO_ROOT)


def test_no_pip_still_routes_through_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    """A uv venv has no pip at all; that branch is unchanged."""
    mod = _rebuild_editable()
    _absent(monkeypatch, "pip")
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/uv" if name == "uv" else None)

    cmd = mod._editable_install_cmd(REPO_ROOT)

    assert cmd[:2] == ["/usr/bin/uv", "pip"]
    assert "--no-build-isolation" not in cmd


def test_neither_pip_nor_uv_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one genuinely unrecoverable combination keeps its named error."""
    mod = _rebuild_editable()
    _absent(monkeypatch, "pip")
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match=r"uv"):
        mod._editable_install_cmd(REPO_ROOT)
