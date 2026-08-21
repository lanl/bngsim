"""A rebuild produces the configuration it was asked for (GH #459).

Every build for one interpreter and platform shares a single CMake build
directory — ``build-dir = "build/{wheel_tag}"``, and the wheel tag records the
Python version and the platform, not the environment being installed into and
not the options passed. So a second install with different options rewrites the
cache ``scripts/rebuild_editable.py`` then builds from.

That script re-specified some options on its configure line and inherited the
rest. ``BNGSIM_ENABLE_KLU`` and ``BNGSIM_REQUIRE_KLU`` were never passed, so they
came entirely from whatever the last build left behind; ``BNGSIM_ENABLE_MIR`` was
passed only when its environment variable was set, so a cached ``ON`` stayed
``ON``. Measured on a normal editable install, one MIR-configured install into a
*separate* venv was enough to turn::

    python -c "import bngsim._bngsim_core as c; print(c.HAS_KLU, c.HAS_MIR)"

from ``True False`` into ``False True`` at the next rebuild, silently: the
staleness guard compares source timestamps against the binary, and the binary
really was fresh — it was just configured differently.

Two halves, both covered here. Nothing configuration-carrying may be left off
the configure line (:data:`_FEATURE_OPTION_DEFAULTS` is the classification, and
``test_every_cmake_option_is_classified`` is what stops a new option from
quietly joining the inherited set). And a cache that disagrees with what the
rebuild is about to ask for stops it, rather than being built on — except where
this invocation asked for the difference itself, which is what
``BNGSIM_ENABLE_MIR=1 python scripts/rebuild_editable.py`` has always meant and
what ``scripts/MIR_VENDORING.md`` documents.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
REBUILD_EDITABLE = REPO_ROOT / "scripts" / "rebuild_editable.py"
CMAKELISTS = REPO_ROOT / "CMakeLists.txt"

pytestmark = pytest.mark.skipif(
    not REBUILD_EDITABLE.exists(),
    reason="scripts/rebuild_editable.py is not in this checkout (installed package)",
)


def _rebuild_editable() -> types.ModuleType:
    """Import the script by path — ``scripts/`` is not a package."""
    name = "_rebuild_editable_feature_options_under_test"
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


def _cmakelists_options() -> dict[str, str]:
    """Every ``option(NAME "doc" DEFAULT)`` in the top-level CMakeLists.txt."""
    # ``\s`` spans newlines on purpose: BNGSIM_ENABLE_LAPACK_DENSE wraps its
    # doc string onto a second line.
    pattern = re.compile(r"""option\(\s*([A-Za-z_][A-Za-z0-9_]*)\s+"[^"]*"\s+(ON|OFF)\s*\)""")
    return dict(pattern.findall(CMAKELISTS.read_text()))


def _defines(cmd: list[str]) -> dict[str, str]:
    """The ``-DNAME=VALUE`` flags of a cmake argv, as a mapping."""
    out: dict[str, str] = {}
    for arg in cmd:
        if arg.startswith("-D") and "=" in arg:
            name, _, value = arg[2:].partition("=")
            out[name] = value
    return out


def _configure_defines(**environ: str) -> dict[str, str]:
    mod = _rebuild_editable()
    settings = mod._resolve_cmake_settings(REPO_ROOT, environ=environ)
    return _defines(
        mod._configure_cmd(
            REPO_ROOT,
            REPO_ROOT / "build" / "whatever",
            settings=settings,
            pybind11_cmake_dir=None,
            sdkroot=None,
            macos_architectures=None,
        )
    )


# ── the classification is complete and matches CMakeLists.txt ────────────────


def test_feature_option_defaults_match_cmakelists() -> None:
    """A default copied here that CMakeLists.txt disagrees with is a silent flip.

    The table is what a rebuild passes when nothing else asks for anything, so a
    stale entry does not fail — it configures the build differently from every
    other route into the same source tree.
    """
    mod = _rebuild_editable()
    declared = _cmakelists_options()
    mismatched = {
        name: (default, declared.get(name))
        for name, default in mod._FEATURE_OPTION_DEFAULTS.items()
        if declared.get(name) != default
    }
    assert not mismatched, (
        "scripts/rebuild_editable.py records a different option() default than "
        f"CMakeLists.txt for {mismatched} (script value, CMakeLists value)."
    )


def test_every_cmake_option_is_classified() -> None:
    """A new option() has to be classified, or it rejoins the inherited set.

    This is the regression that lets GH #459 recur: an option nobody lists is an
    option the configure line does not pass, and an option the configure line
    does not pass is read out of whatever the last build left in the cache.
    """
    mod = _rebuild_editable()
    # BUILD_SHARED_LIBS is not configuration: CMakeLists.txt FORCEs it OFF for
    # every BNGSIM_BUILD_PYTHON build, so the extension is self-contained.
    forced_by_cmakelists = {"BUILD_SHARED_LIBS"}
    classified = (
        set(mod._FEATURE_OPTION_DEFAULTS) | set(mod._PINNED_OPTION_VALUES) | forced_by_cmakelists
    )
    unclassified = set(_cmakelists_options()) - classified
    assert not unclassified, (
        f"CMakeLists.txt declares option(s) {sorted(unclassified)} that "
        "scripts/rebuild_editable.py neither pins nor resolves, so an editable "
        "rebuild inherits them from whichever build last wrote the shared "
        "CMakeCache.txt (GH #459). Add them to _FEATURE_OPTION_DEFAULTS."
    )


# ── the configure line names everything ──────────────────────────────────────


def test_configure_passes_every_feature_option() -> None:
    mod = _rebuild_editable()
    defines = _configure_defines()
    for name in mod._FEATURE_OPTION_DEFAULTS:
        assert name in defines, f"{name} is not on the configure line, so it is inherited"
    for name, value in mod._PINNED_OPTION_VALUES.items():
        assert defines.get(name) == value


def test_configure_passes_the_klu_options_that_used_to_be_inherited() -> None:
    """The two the issue caught: never passed, so entirely the cache's to decide."""
    defines = _configure_defines()
    assert defines["BNGSIM_ENABLE_KLU"] == "ON"
    # pyproject.toml's [tool.scikit-build.cmake.define] requires KLU of every
    # scikit-build-core build; an editable rebuild is not an exception to that.
    assert defines["BNGSIM_REQUIRE_KLU"] == "ON"


def test_configure_pins_the_build_type() -> None:
    """Inheriting CMAKE_BUILD_TYPE is the same bug, one step quieter.

    Nothing in the binary's API changes, so no flag flips and no stub diff
    appears — a tree another install configured ``Debug`` just rebuilds ``Debug``,
    and only the compile lines say so.
    """
    assert _configure_defines()["CMAKE_BUILD_TYPE"] == "Release"


# ── resolution order: CMakeLists default < pyproject < environment ───────────


def test_pyproject_define_beats_the_cmakelists_default() -> None:
    mod = _rebuild_editable()
    assert mod._FEATURE_OPTION_DEFAULTS["BNGSIM_REQUIRE_KLU"] == "OFF"
    settings = mod._resolve_cmake_settings(REPO_ROOT, environ={})
    assert settings["BNGSIM_REQUIRE_KLU"].value == "ON"
    assert settings["BNGSIM_REQUIRE_KLU"].origin == mod._ORIGIN_PYPROJECT


@pytest.mark.parametrize("raw", ["1", "ON", "on", "true", "YES"])
def test_mir_env_var_still_turns_the_backend_on(raw: str) -> None:
    """The spelling scripts/MIR_VENDORING.md documents keeps working."""
    assert _configure_defines(BNGSIM_ENABLE_MIR=raw)["BNGSIM_ENABLE_MIR"] == "ON"


@pytest.mark.parametrize("raw", ["0", "OFF", "off", "false", "no"])
def test_env_var_can_turn_an_option_off(raw: str) -> None:
    """The escape hatch the conflict message points at has to actually work."""
    defines = _configure_defines(BNGSIM_ENABLE_KLU=raw, BNGSIM_REQUIRE_KLU=raw)
    assert defines["BNGSIM_ENABLE_KLU"] == "OFF"
    assert defines["BNGSIM_REQUIRE_KLU"] == "OFF"


def test_empty_env_var_is_not_a_request() -> None:
    """``BNGSIM_ENABLE_MIR=`` is an unset variable that survived an export."""
    mod = _rebuild_editable()
    settings = mod._resolve_cmake_settings(REPO_ROOT, environ={"BNGSIM_ENABLE_MIR": "  "})
    assert settings["BNGSIM_ENABLE_MIR"].value == "OFF"
    assert settings["BNGSIM_ENABLE_MIR"].origin == mod._ORIGIN_DEFAULT


def test_unparseable_env_value_is_refused_by_name() -> None:
    """Silently reading a typo as OFF is how the feature goes missing again."""
    mod = _rebuild_editable()
    with pytest.raises(SystemExit) as excinfo:
        mod._resolve_cmake_settings(REPO_ROOT, environ={"BNGSIM_ENABLE_MIR": "maybe"})
    message = str(excinfo.value)
    assert "BNGSIM_ENABLE_MIR" in message
    assert "maybe" in message


# ── the cache guard ──────────────────────────────────────────────────────────


def _cache(tmp_path: Path, **entries: str) -> Path:
    build_dir = tmp_path / "cp312-cp312-macosx_26_0_arm64"
    build_dir.mkdir()
    lines = ["# This is the CMakeCache file.", "//A comment line", ""]
    for name, value in entries.items():
        kind = "STRING" if name == "CMAKE_BUILD_TYPE" else "BOOL"
        lines.append(f"{name}:{kind}={value}")
        # CMake writes a sibling INTERNAL entry whose name carries a dash; the
        # parser must not read it as a value for the option itself.
        lines.append(f"{name}-ADVANCED:INTERNAL=1")
    (build_dir / "CMakeCache.txt").write_text("\n".join(lines) + "\n")
    return build_dir


def test_cache_values_are_read_and_normalized(tmp_path: Path) -> None:
    mod = _rebuild_editable()
    build_dir = _cache(
        tmp_path,
        BNGSIM_ENABLE_KLU="TRUE",
        BNGSIM_ENABLE_MIR="0",
        CMAKE_BUILD_TYPE="Debug",
    )
    values = mod._cmake_cache_values(
        build_dir, ["BNGSIM_ENABLE_KLU", "BNGSIM_ENABLE_MIR", "CMAKE_BUILD_TYPE"]
    )
    assert values == {
        "BNGSIM_ENABLE_KLU": "ON",
        "BNGSIM_ENABLE_MIR": "OFF",
        "CMAKE_BUILD_TYPE": "Debug",
    }


def test_a_tree_that_was_never_configured_is_not_a_conflict(tmp_path: Path) -> None:
    mod = _rebuild_editable()
    assert mod._cmake_cache_values(tmp_path, ["BNGSIM_ENABLE_KLU"]) == {}


def test_the_reported_contamination_stops_the_rebuild(tmp_path: Path) -> None:
    """The exact tree the issue was filed against: KLU off, MIR on, nobody asking."""
    mod = _rebuild_editable()
    build_dir = _cache(
        tmp_path,
        BNGSIM_ENABLE_KLU="OFF",
        BNGSIM_REQUIRE_KLU="OFF",
        BNGSIM_ENABLE_MIR="ON",
    )
    settings = mod._resolve_cmake_settings(REPO_ROOT, environ={})
    cached = mod._cmake_cache_values(build_dir, settings)
    conflicts = mod._configuration_conflicts(cached, settings)
    assert sorted(conflicts) == sorted(
        [
            ("BNGSIM_ENABLE_KLU", "OFF", "ON"),
            ("BNGSIM_REQUIRE_KLU", "OFF", "ON"),
            ("BNGSIM_ENABLE_MIR", "ON", "OFF"),
        ]
    )

    message = mod._configuration_conflict_message(build_dir, conflicts)
    assert str(build_dir) in message
    for name in ("BNGSIM_ENABLE_KLU", "BNGSIM_REQUIRE_KLU", "BNGSIM_ENABLE_MIR"):
        assert name in message
    # The way out has to be in the message, and it has to be the one that works:
    # reconfiguring in place left an extension that would not load at all.
    assert f"rm -rf {build_dir}" in message
    # ...and so does the way to keep the cached configuration on purpose, spelled
    # so that pasting it does not produce the contradiction CMake fatals on
    # (REQUIRE_KLU=ON with ENABLE_KLU=OFF).
    assert "BNGSIM_ENABLE_KLU=OFF BNGSIM_REQUIRE_KLU=OFF BNGSIM_ENABLE_MIR=ON" in message


def test_an_agreeing_cache_is_not_a_conflict(tmp_path: Path) -> None:
    mod = _rebuild_editable()
    build_dir = _cache(
        tmp_path,
        BNGSIM_ENABLE_KLU="ON",
        BNGSIM_REQUIRE_KLU="ON",
        BNGSIM_ENABLE_MIR="OFF",
        CMAKE_BUILD_TYPE="Release",
    )
    settings = mod._resolve_cmake_settings(REPO_ROOT, environ={})
    cached = mod._cmake_cache_values(build_dir, settings)
    assert mod._configuration_conflicts(cached, settings) == []


def test_a_requested_change_is_not_a_conflict(tmp_path: Path) -> None:
    """Asking for MIR is a reconfigure, not drift — the documented workflow."""
    mod = _rebuild_editable()
    build_dir = _cache(tmp_path, BNGSIM_ENABLE_MIR="OFF")
    settings = mod._resolve_cmake_settings(REPO_ROOT, environ={"BNGSIM_ENABLE_MIR": "1"})
    cached = mod._cmake_cache_values(build_dir, settings)
    assert mod._configuration_conflicts(cached, settings) == []


def test_a_debug_cache_stops_a_release_rebuild(tmp_path: Path) -> None:
    mod = _rebuild_editable()
    build_dir = _cache(tmp_path, CMAKE_BUILD_TYPE="Debug")
    settings = mod._resolve_cmake_settings(REPO_ROOT, environ={})
    cached = mod._cmake_cache_values(build_dir, settings)
    assert mod._configuration_conflicts(cached, settings) == [
        ("CMAKE_BUILD_TYPE", "Debug", "Release")
    ]


def test_an_undecided_cache_entry_is_not_a_conflict(tmp_path: Path) -> None:
    """CMake writes a bare ``CMAKE_BUILD_TYPE:STRING=`` into a hand-configured tree.

    That is a setting the tree never decided, not one it disagrees about, and the
    configure line pins it either way. Reading it as a disagreement would refuse
    to rebuild in a tree with nothing wrong with it — and would say so with an
    empty value on one side of the comparison.
    """
    mod = _rebuild_editable()
    build_dir = _cache(tmp_path, CMAKE_BUILD_TYPE="")
    settings = mod._resolve_cmake_settings(REPO_ROOT, environ={})
    cached = mod._cmake_cache_values(build_dir, settings)
    assert "CMAKE_BUILD_TYPE" not in cached
    assert mod._configuration_conflicts(cached, settings) == []


def test_a_cache_without_the_entry_is_not_a_conflict(tmp_path: Path) -> None:
    """A partially written cache must not be read as a disagreement."""
    mod = _rebuild_editable()
    build_dir = _cache(tmp_path, BNGSIM_ENABLE_KLU="ON")
    settings = mod._resolve_cmake_settings(REPO_ROOT, environ={})
    cached = mod._cmake_cache_values(build_dir, settings)
    assert mod._configuration_conflicts(cached, settings) == []


# ── the stub the drift used to land in ───────────────────────────────────────


def test_capability_flags_are_normalized_out_of_the_stub() -> None:
    """``HAS_*`` describes one build, and the committed stub is everyone's.

    This is where the issue was noticed: a rebuild that inherited another
    configuration rewrote ``HAS_KLU``/``HAS_MIR`` in a tracked file, and
    committing that by accident would have been easy. It is the same hazard the
    ``__build_commit__`` stamp already had, so it gets the same treatment — and
    it also covers the deliberate case, where a developer builds with
    ``-DBNGSIM_ENABLE_MIR=ON`` and should not be left holding a diff.
    """
    mod = _rebuild_editable()
    generated = "HAS_KLU: bool = True\nHAS_MIR: bool = False\nHAS_NEWTHING: bool = True\n"
    assert mod._normalize_stub_build_stamps(generated) == (
        "HAS_KLU: bool = ...\nHAS_MIR: bool = ...\nHAS_NEWTHING: bool = ...\n"
    )


def test_the_committed_stub_carries_no_capability_values() -> None:
    stub = REPO_ROOT / "python" / "bngsim" / "_bngsim_core.pyi"
    if not stub.is_file():
        pytest.skip(f"no committed stub at {stub}")
    valued = [
        line
        for line in stub.read_text().splitlines()
        if re.match(r"^HAS_[A-Z0-9_]*: bool = (?:True|False)$", line)
    ]
    assert not valued, (
        f"the committed stub records one build's capability flags ({valued}). They "
        "flip with the build configuration, so a rebuild elsewhere reverses the "
        "diff. Regenerate through scripts/rebuild_editable.py, which normalizes "
        "them to `...`."
    )
