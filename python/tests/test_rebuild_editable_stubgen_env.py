"""`scripts/rebuild_editable.py` must let pybind11-stubgen import what it built.

`main()` lists its build dir in `SKBUILD_EDITABLE_SKIP` so that a child that
imports bngsim mid-build starts no rebuild of its own, and it handed the same
environment to the stub step. scikit-build-core's editable finder (since its
#1612) stops resolving the package at all for a build dir listed there, so after
a build and install that had succeeded the rebuild died in the stub step with
"No module named 'bngsim._bngsim_core'". The stub step now runs without that
entry, and keeps any the caller set.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REBUILD_EDITABLE = REPO_ROOT / "scripts" / "rebuild_editable.py"

pytestmark = pytest.mark.skipif(
    not REBUILD_EDITABLE.exists(),
    reason="scripts/rebuild_editable.py is not in this checkout (installed package)",
)


def _load_rebuild_editable() -> types.ModuleType:
    """Import rebuild_editable.py by path — scripts/ is not a package."""
    name = "_rebuild_editable_stubgen_env_under_test"
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
def stubgen_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, dict[str, str]]:
    """Run `main()` with every subprocess recorded, and keep the environment the
    stub generator is given. The stub lands in a scratch tree, not the checkout."""
    mod = _load_rebuild_editable()
    seen: dict[str, dict[str, str]] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> None:
        if "pybind11_stubgen" in cmd:
            seen["env"] = dict(kwargs["env"])  # type: ignore[arg-type]
            out = Path(cmd[cmd.index("--output-dir") + 1]) / "bngsim"
            out.mkdir(parents=True)
            (out / "_bngsim_core.pyi").write_text("")

    real_find_spec = importlib.util.find_spec

    def find_spec(name: str, *args: object, **kwargs: object) -> object:
        if name == "pybind11_stubgen":
            return object()
        return real_find_spec(name, *args, **kwargs)  # type: ignore[arg-type]

    real_stub = mod._regenerate_stub
    scratch = tmp_path / "src"
    (scratch / "python" / "bngsim").mkdir(parents=True)

    monkeypatch.setattr(mod, "_run", fake_run)
    monkeypatch.setattr(mod, "_load_build_info", lambda src: {"build_dir": str(tmp_path / "bld")})
    monkeypatch.setattr(mod, "_install_prefix", lambda: tmp_path / "prefix")
    monkeypatch.setattr(mod.importlib.util, "find_spec", find_spec)
    monkeypatch.setattr(mod, "_regenerate_stub", lambda source_dir, **kw: real_stub(scratch, **kw))
    monkeypatch.setenv("BNGSIM_SKIP_METADATA_REFRESH", "1")
    monkeypatch.delenv("BNGSIM_SKIP_STUBGEN", raising=False)
    return seen


def test_the_stub_step_is_not_told_to_skip_the_build_it_imports(
    stubgen_env: dict[str, dict[str, str]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SKBUILD_EDITABLE_SKIP", raising=False)
    mod = _load_rebuild_editable()

    assert mod.main() == 0

    assert "SKBUILD_EDITABLE_SKIP" not in stubgen_env["env"]


def test_the_stub_step_keeps_a_skip_entry_the_caller_set(
    stubgen_env: dict[str, dict[str, str]], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Only the rebuild's own entry goes; one the caller set stays."""
    other = str(tmp_path / "another-build")
    monkeypatch.setenv("SKBUILD_EDITABLE_SKIP", other)
    mod = _load_rebuild_editable()

    assert mod.main() == 0

    assert stubgen_env["env"]["SKBUILD_EDITABLE_SKIP"] == other
    assert str((tmp_path / "bld").resolve()) not in stubgen_env["env"][
        "SKBUILD_EDITABLE_SKIP"
    ].split(os.pathsep)
