"""`scripts/local_ci.py` must not build a configuration pyproject forbids.

The local matrix is what stands in for GitHub Actions while it is paused
(`scripts/LOCAL_CI.md`), so the wheels it builds are the evidence a change is
safe. It sets a few things in the build environment itself, and those additions
sit *next to* `[tool.scikit-build.cmake.define]` in pyproject, which reaches
every scikit-build-core build — nothing checked the two against each other.

They diverged. `build_wheel` passed `CMAKE_ARGS=-DBNGSIM_ENABLE_KLU=OFF` on
macOS from the initial release; `BNGSIM_REQUIRE_KLU = "ON"` was added to
pyproject months later (GH #209, "no silent dense-only fallback"). CMake refuses
that combination outright — `-DBNGSIM_REQUIRE_KLU=ON contradicts
-DBNGSIM_ENABLE_KLU=OFF` — so from then on `local_ci.py matrix` failed to
configure for all four Pythons on macOS and wrote `build: FAIL` into the report.
And before that it was quietly validating a dense-only wheel that no published
macOS wheel resembles, since `[tool.cibuildwheel.macos]` sets both to `ON`.

So the rule pinned here is the one CMake already enforces, checked at the level
where the two halves are written: whatever the matrix adds to a build must not
switch off a feature pyproject requires.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
import textwrap
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib", reason="tomllib is 3.11+")

REPO_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = REPO_ROOT / "pyproject.toml"
LOCAL_CI = REPO_ROOT / "scripts" / "local_ci.py"

pytestmark = pytest.mark.skipif(
    not LOCAL_CI.exists(),
    reason="scripts/local_ci.py is not in this checkout (installed package)",
)


def _load_local_ci():
    """Import local_ci.py by path — scripts/ is not a package."""
    name = "_local_ci_build_env_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, LOCAL_CI)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules[name]
        raise
    return mod


def _required_features() -> list[str]:
    """`BNGSIM_REQUIRE_<X>=ON` defines pyproject applies to every build."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    defines = data["tool"]["scikit-build"]["cmake"]["define"]
    return [
        name
        for name, value in defines.items()
        if name.startswith("BNGSIM_REQUIRE_") and str(value).upper() == "ON"
    ]


def test_pyproject_requires_klu_for_every_build() -> None:
    """The premise of the test below, asserted rather than assumed (GH #209)."""
    assert "BNGSIM_REQUIRE_KLU" in _required_features(), (
        "pyproject no longer requires KLU for every scikit-build-core build; "
        "if that was deliberate, the contradiction check below has lost its subject"
    )


def _string_literals(func) -> list[str]:
    """Every string literal in ``func``'s body.

    The flags in question are literals in the ``CMAKE_ARGS`` assignment, and
    reading the raw source instead would match the comment explaining why the
    flag is gone — a test that cannot tell an argument from a note about it.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    return [
        n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]


def test_matrix_build_never_disables_a_feature_pyproject_requires() -> None:
    """`REQUIRE_X=ON` plus `ENABLE_X=OFF` is a fatal error in CMakeLists.txt.

    Written against every `BNGSIM_REQUIRE_*` pyproject declares, not KLU alone:
    the next feature to get a hard requirement inherits the check instead of
    re-learning it the way this one did.
    """
    mod = _load_local_ci()
    literals = _string_literals(mod.build_wheel)

    for required in _required_features():
        disabled = required.replace("_REQUIRE_", "_ENABLE_") + "=OFF"
        offenders = [text for text in literals if disabled in text]
        assert not offenders, (
            f"local_ci.py's build env passes -D{disabled} ({offenders}) while pyproject "
            f"sets {required}=ON for every build; CMake refuses that combination, so the "
            "matrix cannot configure at all on the platform that sets it."
        )
