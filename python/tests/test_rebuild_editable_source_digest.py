"""Issue #528: ``scripts/rebuild_editable.py`` leaves the record that lets the
stale-binary guard see through a rewrite that leaves the C++ byte-identical.

The guard (``bngsim._build_provenance``, issue #125) calls a binary STALE when a
watched source is newer than it. The routine end of a C++ pull request —
``git checkout main && git pull`` after a squash merge — rewrites every touched
source with the bytes it already had, and the guard then demanded a rebuild of a
binary that was current. The rebuild script now digests the watched sources under
its lock before the configure and again after the install, and when the two agree
writes a record beside the installed extension: the extension's SHA-256 and that
digest. ``test_build_provenance.py`` pins how the guard reads the record; this file
pins the writer — what it writes, when it withholds the record, and that ``main()``
takes the two digests around the build and nowhere else.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _source_root import bngsim_source_root  # noqa: E402

# The provenance module is loaded BY PATH, never imported as bngsim._build_
# provenance -- test_the_guard_loads_without_importing_bngsim asserts exactly
# that. So reaching the real source tree here cannot shadow anything, and the
# rig's env vars are how to reach it: a __file__ walk-up lands in run_tests.sh's
# stand-in, where python/ holds tests/ and nothing else, and skipped all nine of
# these tests (issue #594).
REPO_ROOT = bngsim_source_root() or Path(__file__).resolve().parents[2]
REBUILD_EDITABLE = REPO_ROOT / "scripts" / "rebuild_editable.py"
#: What every test here ultimately loads: ``_load_build_provenance`` reads it out
#: of the source tree by path. The guard named only the script until issue #590,
#: so a tree carrying scripts/ but not the package -- a wheel or subtree checkout
#: -- passed it and then died on a FileNotFoundError, eight times.
PROVENANCE = REPO_ROOT / "python" / "bngsim" / "_build_provenance.py"

pytestmark = pytest.mark.skipif(
    not (REBUILD_EDITABLE.exists() and PROVENANCE.exists()),
    reason="scripts/rebuild_editable.py or python/bngsim/_build_provenance.py "
    "is not in this checkout (installed package)",
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
def mod() -> types.ModuleType:
    return _load_rebuild_editable()


@pytest.fixture
def guard(mod: types.ModuleType) -> types.ModuleType:
    """The guard module as the script loads it: by path, not through ``bngsim``."""
    return mod._load_build_provenance(REPO_ROOT)


def _tree(tmp_path: Path) -> tuple[Path, Path]:
    """A two-file source tree and an installed stand-in extension."""
    root = tmp_path / "checkout"
    (root / "src").mkdir(parents=True)
    (root / "CMakeLists.txt").write_text("project(x)\n")
    (root / "src" / "model.cpp").write_text("int rate() { return 1; }\n")
    extension = tmp_path / "prefix" / "_bngsim_core.cpython-312-darwin.so"
    extension.parent.mkdir(parents=True)
    extension.write_bytes(b"compiled from int rate() { return 1; }")
    return root, extension


# ── What the record step writes, and when it refuses ─────────────────────────


def test_sources_that_held_still_leave_a_record_the_guard_accepts(
    mod: types.ModuleType,
    guard: types.ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, extension = _tree(tmp_path)

    status = mod._record_source_digest(guard, root, extension, guard.source_digest(root))

    assert status.startswith("written (")
    assert f"source_digest_record={status}" in capsys.readouterr().out
    assert guard._record_vouches(extension, root)
    # The rewrite the issue is about: the same bytes again, and the record still holds.
    source = root / "src" / "model.cpp"
    source.write_bytes(source.read_bytes())
    assert guard._record_vouches(extension, root)
    source.write_bytes(source.read_bytes() + b" ")
    assert not guard._record_vouches(extension, root)


def test_sources_that_moved_during_the_rebuild_leave_no_record(
    mod: types.ModuleType, guard: types.ModuleType, tmp_path: Path
) -> None:
    """The first digest no longer describes what the build may have read, so no
    record — and an earlier rebuild's record does not survive to speak for this one."""
    root, extension = _tree(tmp_path)
    guard.write_source_record(extension, guard.source_digest(root))
    before = guard.source_digest(root)
    (root / "src" / "model.cpp").write_text("int rate() { return 2; }\n")  # an edit mid-build

    status = mod._record_source_digest(guard, root, extension, before)

    assert "changed during the rebuild" in status
    assert not guard.source_record_path(extension).exists()


def test_sources_that_could_not_be_hashed_leave_no_record(
    mod: types.ModuleType, guard: types.ModuleType, tmp_path: Path
) -> None:
    root, extension = _tree(tmp_path)
    guard.write_source_record(extension, guard.source_digest(root))

    status = mod._record_source_digest(guard, root, extension, None)

    assert "could not be hashed" in status
    assert not guard.source_record_path(extension).exists()


def test_no_installed_extension_no_record(
    mod: types.ModuleType, guard: types.ModuleType, tmp_path: Path
) -> None:
    root, extension = _tree(tmp_path)
    extension.unlink()

    status = mod._record_source_digest(guard, root, extension, guard.source_digest(root))

    assert status.startswith("skipped (no installed extension")
    assert not guard.source_record_path(extension).exists()


def test_a_record_that_cannot_be_written_costs_the_record_not_the_rebuild(
    mod: types.ModuleType,
    guard: types.ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, extension = _tree(tmp_path)

    def read_only(*_a: object, **_k: object) -> Path:
        raise PermissionError("read-only site-packages")

    monkeypatch.setattr(guard, "write_source_record", read_only)

    status = mod._record_source_digest(guard, root, extension, guard.source_digest(root))

    assert status.startswith("skipped (the record could not be updated")


def test_the_guard_loads_without_importing_bngsim() -> None:
    """Importing ``bngsim`` would load the ``_bngsim_core`` this script is about to
    replace, and run the import-time guard against it. A fresh interpreter shows it
    does neither."""
    code = "\n".join(
        [
            "import importlib.util, sys",
            "from pathlib import Path",
            f"script = Path({str(REBUILD_EDITABLE)!r})",
            "spec = importlib.util.spec_from_file_location('rebuild_editable', script)",
            "mod = importlib.util.module_from_spec(spec)",
            "sys.modules['rebuild_editable'] = mod",
            "spec.loader.exec_module(mod)",
            "guard = mod._load_build_provenance(script.parents[1])",
            "assert guard.source_digest(script.parents[1])",
            "print(sorted(m for m in sys.modules if m == 'bngsim' or m.startswith('bngsim.')))",
        ]
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", out.stdout


# ── Where main() takes the two digests ───────────────────────────────────────


@pytest.fixture
def driven_main(
    mod: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> types.SimpleNamespace:
    """Run ``main()`` with every subprocess recorded instead of run, and the install
    step leaving an extension behind, so the record step has something to name."""
    events: list[str] = []
    prefix = tmp_path / "prefix"
    extension = prefix / f"_bngsim_core{mod._ext_suffix()}"

    def fake_run(cmd: list[str], **_kwargs: object) -> None:
        if cmd[:1] == ["cmake"] and "-S" in cmd:
            events.append("configure")
        elif cmd[:2] == ["cmake", "--build"]:
            events.append("build")
        elif cmd[:2] == ["cmake", "--install"]:
            prefix.mkdir(parents=True, exist_ok=True)
            extension.write_bytes(b"installed extension")
            events.append("install")

    monkeypatch.setattr(mod, "_run", fake_run)
    monkeypatch.setattr(mod, "_load_build_info", lambda src: {"build_dir": str(tmp_path / "bld")})
    monkeypatch.setattr(mod, "_install_prefix", lambda: prefix)
    monkeypatch.setattr(mod, "_regenerate_stub", lambda *a, **k: None)
    monkeypatch.setenv("BNGSIM_SKIP_METADATA_REFRESH", "1")
    return types.SimpleNamespace(events=events, extension=extension)


def _logging_guard(events: list[str], digests: list[str]) -> types.SimpleNamespace:
    """A stand-in guard that logs each call in among the build steps."""
    pending = list(digests)

    def source_digest(_root: Path) -> str:
        events.append("digest")
        return pending.pop(0)

    def write_source_record(extension: Path, sources: str) -> Path:
        events.append(f"write {sources}")
        return extension

    def remove_source_record(_extension: Path) -> bool:
        events.append("remove")
        return False

    return types.SimpleNamespace(
        source_digest=source_digest,
        write_source_record=write_source_record,
        remove_source_record=remove_source_record,
    )


def test_main_digests_before_the_configure_and_after_the_install(
    mod: types.ModuleType, driven_main: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard = _logging_guard(driven_main.events, ["same", "same"])
    monkeypatch.setattr(mod, "_load_build_provenance", lambda _src: guard)

    assert mod.main() == 0

    assert driven_main.events == [
        "digest",
        "configure",
        "build",
        "install",
        "digest",
        "write same",
    ]


def test_main_withholds_the_record_when_the_digests_disagree(
    mod: types.ModuleType, driven_main: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    guard = _logging_guard(driven_main.events, ["before", "after"])
    monkeypatch.setattr(mod, "_load_build_provenance", lambda _src: guard)

    assert mod.main() == 0

    assert driven_main.events == ["digest", "configure", "build", "install", "digest", "remove"]


def test_main_leaves_a_record_the_guard_accepts_for_this_checkout(
    mod: types.ModuleType, driven_main: types.SimpleNamespace
) -> None:
    """End to end with the real guard module over this checkout's own sources: the
    record ``main()`` writes is one the import-time guard accepts."""
    assert mod.main() == 0

    guard = mod._load_build_provenance(REPO_ROOT)
    assert guard.source_record_path(driven_main.extension).is_file()
    assert guard._record_vouches(driven_main.extension, REPO_ROOT)
