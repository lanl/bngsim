"""The swig AMICI is built with is the swig uv.lock installs (issue #1824).

AMICI builds from source in an isolated env and refuses to compile a model
unless the swig in the venv is the one it was built with. pyproject.toml pins
the build's swig through ``[tool.uv] build-constraint-dependencies``; a swig
bump in uv.lock that left the pin behind would fail every amici-sens reference
of the weekly live parity run, and nothing else.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib", reason="tomllib is 3.11+")

ROOT = Path(__file__).resolve().parents[2]


def _locked(name: str) -> str:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    (version,) = {p["version"] for p in lock["package"] if p["name"] == name}
    return version


def test_the_build_constraint_on_swig_is_the_locked_swig() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pins = project["tool"]["uv"]["build-constraint-dependencies"]
    (swig,) = [p for p in pins if re.match(r"swig\s*==", p)]
    assert swig.split("==", 1)[1].strip() == _locked("swig")


def test_uv_lock_records_the_same_constraint() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    constraints = {c["name"]: c["specifier"] for c in lock["manifest"]["build-constraints"]}
    assert constraints["swig"] == f"=={_locked('swig')}"
