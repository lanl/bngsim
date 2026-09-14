"""Regression net for the stale-binary guard (issue #125).

The editable install loads a separately-built ``_bngsim_core`` that does not
auto-rebuild (``editable.rebuild = false``, #23). A forgotten rebuild makes the
suite report on OLD C++ — a green run that gets committed as a correctness
verdict about code that isn't running (the GH #118 near-miss). ``_build_provenance``
turns that invisible failure into a loud one; these tests pin its behaviour.

The verdict is decided by mtime, with a source-digest record as the tiebreaker
when the mtime says stale (issue #528) — still a passive check that must never
shell out to ninja/cmake, per #23. So the logic is exercised with real files on
disk at controlled mtimes, not mocks of the filesystem.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import replace
from pathlib import Path

import pytest
from bngsim import _build_provenance as bp

# ── The live-environment invariant ───────────────────────────────────────────
# This is the meta-test: the binary running this very suite must be in sync with
# its source. The conftest preflight already enforces this before collection;
# asserting it here documents the invariant and fails loudly if anyone bypasses
# the preflight (e.g. imports a test module directly).


def test_loaded_binary_is_not_stale() -> None:
    """The binary under test must reflect current C++ (or the suite is lying)."""
    if os.environ.get("BNGSIM_ALLOW_STALE_CORE") or os.environ.get("BNGSIM_NO_BUILD_CHECK"):
        pytest.skip("stale-binary check explicitly bypassed via env")
    prov = bp.gather()
    if not prov.is_source_checkout:
        pytest.skip("installed wheel (no source tree) — guard is a no-op here")
    assert not prov.is_stale, bp.format_report(prov)


def test_build_commit_stamp_present() -> None:
    """A freshly built editable core exposes a non-'unknown' __build_commit__."""
    from bngsim import _bngsim_core

    commit = getattr(_bngsim_core, "__build_commit__", None)
    assert commit is not None, "__build_commit__ missing — CMake stamp (issue #125) not wired"
    if not bp.gather().is_source_checkout:
        return  # an installed wheel may legitimately carry 'unknown'
    assert commit != "unknown", (
        "core built without git provenance; rebuild from a git checkout so "
        "__build_commit__ records the source commit (issue #125)."
    )


def test_pybind11_version_stamp_present() -> None:
    """The binary must say which pybind11 built it (issue #288).

    It said nothing before, in the extension or the wheel, so a build that
    resolved pybind11 from an unrelated interpreter's site-packages instead of
    the one its own `[build-system] requires` installed produced an artifact
    indistinguishable from the intended one. Two wheels from a single commit,
    built for two different interpreters, were both silently compiled against a
    third environment's copy — and nothing anywhere recorded it.
    """
    from bngsim import _bngsim_core

    stamp = getattr(_bngsim_core, "__pybind11_version__", None)
    assert stamp is not None, (
        "__pybind11_version__ missing — the CMake stamp (issue #288) is not wired"
    )
    assert stamp not in ("", "unknown"), (
        "the extension was built without find_package reporting a pybind11 version; "
        "which pybind11 compiled this binary is exactly what #288 made recordable."
    )
    assert stamp[0].isdigit(), f"expected a version string, got {stamp!r}"


@pytest.mark.parametrize("attr", ["__build_commit__", "__pybind11_version__"])
def test_committed_stub_carries_no_build_stamp(attr: str) -> None:
    """The *committed* stub must NOT carry any one build's provenance.

    The runtime attributes above are per-build and must be real. The ``.pyi`` is
    the opposite: pybind11-stubgen copies whatever the just-built module reports,
    so ``scripts/rebuild_editable.py`` — which every C++ change runs — rewrites
    those lines into a committed file on every invocation. Left alone it
    produces a spurious diff per rebuild and bakes in one developer's commit, or
    a ``+dirty`` marker from a half-finished tree. PR #70 merged
    ``'e61f83d57358+dirty'`` exactly that way. ``__pybind11_version__`` (#288) is
    the same hazard: it differs between two developers whose environments
    resolved different pybind11 versions, and would flip back and forth in the
    committed file — visibility belongs in the *binary*, not in a diff.

    ``_normalize_stub_build_stamps`` in that script pins both to ``'unknown'``
    (CMake's own no-provenance default); this is the check that notices if the
    normalization is ever dropped or bypassed. Nothing reads the stub's value —
    mypy checks the declared *type* — so pinning it costs nothing.
    """
    prov = bp.gather()
    if not prov.is_source_checkout:
        pytest.skip("installed wheel (no source tree) — no committed stub to check")

    # Resolve via the imported package, not __file__: run_tests.sh copies the
    # test modules to a temp dir, which would strand a __file__-relative path.
    # The editable install points bngsim.__file__ at python/bngsim/__init__.py,
    # so the committed stub is its sibling.
    import bngsim

    stub = Path(bngsim.__file__).resolve().parent / "_bngsim_core.pyi"
    assert stub.is_file(), f"committed stub missing from the source package ({stub})"

    stamps = [line for line in stub.read_text().splitlines() if line.startswith(f"{attr}: str = ")]
    assert len(stamps) == 1, f"expected exactly one {attr} line, got {stamps}"
    assert stamps[0] == f"{attr}: str = 'unknown'", (
        f"committed stub carries a machine-specific build stamp ({stamps[0]!r}). "
        "Re-run scripts/rebuild_editable.py, or normalize the line by hand — the "
        "stub must not record what any particular build came from."
    )


# ── Staleness verdict logic ───────────────────────────────────────────────────


def _prov(core_mtime: float | None, newest_mtime: float | None) -> bp.Provenance:
    """A Provenance with only the two fields the staleness verdict reads."""
    return bp.Provenance(
        core_path=Path("/fake/_bngsim_core.so"),
        core_mtime=core_mtime,
        build_commit="deadbeef",
        source_root=Path("/fake/root"),
        newest_source=Path("/fake/root/src/x.cpp"),
        newest_source_mtime=newest_mtime,
        head_commit="deadbeef",
    )


def test_is_stale_true_when_source_newer_beyond_slack() -> None:
    assert _prov(core_mtime=1000.0, newest_mtime=1000.0 + bp._MTIME_SLACK + 5).is_stale


def test_is_stale_false_within_slack() -> None:
    # A source file a hair newer than the binary (same-second build/copy jitter)
    # must NOT trip the guard.
    assert not _prov(core_mtime=1000.0, newest_mtime=1000.0 + bp._MTIME_SLACK / 2).is_stale


def test_is_stale_false_when_binary_newer() -> None:
    assert not _prov(core_mtime=2000.0, newest_mtime=1000.0).is_stale


def test_is_stale_false_when_mtime_unknown() -> None:
    assert not _prov(core_mtime=None, newest_mtime=2000.0).is_stale
    assert not _prov(core_mtime=1000.0, newest_mtime=None).is_stale


def test_installed_wheel_provenance_is_noop() -> None:
    """No source root → not a checkout → never stale (end-user wheel path)."""
    prov = bp.Provenance(
        core_path=Path("/site-packages/bngsim/_bngsim_core.so"),
        core_mtime=1000.0,
        build_commit="deadbeef",
        source_root=None,
        newest_source=None,
        newest_source_mtime=None,
        head_commit=None,
    )
    assert not prov.is_source_checkout
    assert not prov.is_stale


# ── _newest_source: real files, real mtimes ───────────────────────────────────


def test_newest_source_finds_latest_cpp(tmp_path: Path) -> None:
    root = tmp_path
    cmake = root / "CMakeLists.txt"
    cmake.write_text("project(x)")
    os.utime(cmake, (500.0, 500.0))  # CMakeLists.txt is scanned too — pin it old
    src = root / "src"
    src.mkdir()
    inc = root / "include" / "bngsim"
    inc.mkdir(parents=True)
    tp = root / "third_party" / "nfsim"
    tp.mkdir(parents=True)

    old = src / "old.cpp"
    old.write_text("// old")
    os.utime(old, (1000.0, 1000.0))
    header = inc / "model.hpp"
    header.write_text("// h")
    os.utime(header, (1500.0, 1500.0))
    newest = tp / "patched.cpp"  # a vendored hand-patch (e.g. the #116 fix)
    newest.write_text("// new")
    os.utime(newest, (2000.0, 2000.0))

    path, mtime = bp._newest_source(root)
    assert path == newest
    assert mtime == pytest.approx(2000.0)


def test_newest_source_counts_the_projects_cmake_modules(tmp_path: Path) -> None:
    """A build-graph edit changes the binary even when no C++ moved (GH #288).

    `cmake/BngsimResolvePybind11.cmake` decides which pybind11 headers the
    extension compiles against. Watching only `CMakeLists.txt` and C++ would call
    a binary fresh after exactly the kind of edit that changes what it is.
    """
    root = tmp_path
    (root / "CMakeLists.txt").write_text("project(x)")
    os.utime(root / "CMakeLists.txt", (500.0, 500.0))
    src = root / "src"
    src.mkdir()
    code = src / "a.cpp"
    code.write_text("// code")
    os.utime(code, (1000.0, 1000.0))
    modules = root / "cmake"
    modules.mkdir()
    newest = modules / "BngsimResolvePybind11.cmake"
    newest.write_text("# module")
    os.utime(newest, (2000.0, 2000.0))

    path, mtime = bp._newest_source(root)
    assert path == newest
    assert mtime == pytest.approx(2000.0)


def test_newest_source_ignores_non_source_files(tmp_path: Path) -> None:
    root = tmp_path
    src = root / "src"
    src.mkdir()
    code = src / "a.cpp"
    code.write_text("// code")
    os.utime(code, (1000.0, 1000.0))
    # A newer NON-source file (build log, README) must not count as source.
    noise = src / "notes.txt"
    noise.write_text("notes")
    os.utime(noise, (9000.0, 9000.0))

    path, mtime = bp._newest_source(root)
    assert path == code
    assert mtime == pytest.approx(1000.0)


def test_newest_source_empty_tree_returns_none(tmp_path: Path) -> None:
    path, mtime = bp._newest_source(tmp_path)
    assert path is None and mtime is None


# ── Env gates ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value,expected", [("1", True), ("true", True), ("", False), ("0", False)]
)
def test_no_build_check_gate(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool) -> None:
    monkeypatch.setenv("BNGSIM_NO_BUILD_CHECK", value)
    assert bp._checks_disabled() is expected


def test_is_stale_respects_disable_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bp, "gather", lambda **_: _prov(core_mtime=0.0, newest_mtime=1e9))
    monkeypatch.setenv("BNGSIM_NO_BUILD_CHECK", "1")
    assert bp.is_stale() is False
    monkeypatch.delenv("BNGSIM_NO_BUILD_CHECK")
    assert bp.is_stale() is True


# ── enforce() / warn_if_stale(): the teeth ────────────────────────────────────


def test_enforce_raises_on_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bp, "gather", lambda **_: _prov(core_mtime=0.0, newest_mtime=1e9))
    monkeypatch.delenv("BNGSIM_ALLOW_STALE_CORE", raising=False)
    monkeypatch.delenv("BNGSIM_NO_BUILD_CHECK", raising=False)
    with pytest.raises(bp.StaleBinaryError):
        bp.enforce()


def test_enforce_downgrades_to_warning_when_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bp, "gather", lambda **_: _prov(core_mtime=0.0, newest_mtime=1e9))
    monkeypatch.setenv("BNGSIM_ALLOW_STALE_CORE", "1")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bp.enforce()
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_enforce_noop_when_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bp, "gather", lambda **_: _prov(core_mtime=2000.0, newest_mtime=1000.0))
    monkeypatch.delenv("BNGSIM_ALLOW_STALE_CORE", raising=False)
    bp.enforce()  # must not raise


def test_warn_if_stale_emits_and_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bp, "gather", lambda **_: _prov(core_mtime=0.0, newest_mtime=1e9))
    monkeypatch.delenv("BNGSIM_NO_BUILD_CHECK", raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bp.warn_if_stale()
    assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_warn_if_stale_swallows_internal_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(**_):
        raise RuntimeError("gather blew up")

    monkeypatch.setattr(bp, "gather", _boom)
    # A provenance hiccup must never be the reason `import bngsim` fails.
    bp.warn_if_stale()


# ── blocking_report(): the pytest-preflight decision ──────────────────────────


def test_blocking_report_blocks_on_stale(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BNGSIM_ALLOW_STALE_CORE", raising=False)
    monkeypatch.delenv("BNGSIM_NO_BUILD_CHECK", raising=False)
    report = bp.blocking_report(_prov(core_mtime=0.0, newest_mtime=1e9))
    assert report is not None and "STALE" in report


def test_blocking_report_none_when_fresh() -> None:
    assert bp.blocking_report(_prov(core_mtime=2000.0, newest_mtime=1000.0)) is None


def test_blocking_report_none_when_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BNGSIM_ALLOW_STALE_CORE", "1")
    assert bp.blocking_report(_prov(core_mtime=0.0, newest_mtime=1e9)) is None


def test_blocking_report_none_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BNGSIM_NO_BUILD_CHECK", "1")
    assert bp.blocking_report(_prov(core_mtime=0.0, newest_mtime=1e9)) is None


# ── Reporting strings ─────────────────────────────────────────────────────────


def test_format_report_flags_stale_and_points_to_rebuild() -> None:
    report = bp.format_report(_prov(core_mtime=0.0, newest_mtime=1e9))
    assert "STALE" in report
    assert "rebuild_editable.py" in report


def test_stale_report_flags_a_rebuild_prerequisite_this_env_lacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remedy that fails sends the reader to BNGSIM_ALLOW_STALE_CORE=1 (GH #229).

    ``scripts/rebuild_editable.py`` drives cmake against this environment, so it
    needs pybind11 here — and pybind11 lives only in ``[build-system] requires``,
    which uv never installs into ``.venv``. When it is missing, say so *next to*
    the remedy, because the alternative sitting beside that remedy is the escape
    hatch the whole guard exists to keep people off.
    """
    monkeypatch.setattr(bp, "_pybind11_missing", lambda: True)
    report = bp.format_report(_prov(core_mtime=0.0, newest_mtime=1e9))
    assert "pybind11" in report
    assert "uv sync --extra dev" in report


def test_stale_report_stays_quiet_when_pybind11_is_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It is a conditional aside, not boilerplate: no note when nothing is missing."""
    monkeypatch.setattr(bp, "_pybind11_missing", lambda: False)
    report = bp.format_report(_prov(core_mtime=0.0, newest_mtime=1e9))
    assert "pybind11" not in report
    assert "rebuild_editable.py" in report


def test_pybind11_probe_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A diagnostic aside must not be able to break the guard it annotates."""

    def boom(name: str) -> None:
        raise RuntimeError("broken import system")

    monkeypatch.setattr(bp.sys.modules["importlib.util"], "find_spec", boom)
    assert bp._pybind11_missing() is False


def test_identity_line_is_single_line() -> None:
    line = bp.identity_line(_prov(core_mtime=2000.0, newest_mtime=1000.0))
    assert "\n" not in line
    assert "_bngsim_core" in line


# ── The source-digest record (issue #528) ─────────────────────────────────────
#
# An mtime comparison calls a current binary STALE after anything rewrites a
# watched file with the bytes it already had: `git checkout main && git pull`
# after a squash merge, `git stash` / `stash pop`, a rebase. scripts/rebuild_editable.py
# leaves a record beside the extension it installed (the extension's SHA-256 and a
# digest of the watched sources), and gather() reads it only when the mtime says
# stale. Real files at controlled mtimes again; only the lookups of the loaded
# binary and its source root are pointed at the test tree. The writer's side is
# pinned in test_rebuild_editable_source_digest.py.


def _checkout(tmp_path: Path) -> tuple[Path, Path]:
    """A two-file source tree and a stand-in extension built after both sources."""
    root = tmp_path / "checkout"
    (root / "src").mkdir(parents=True)
    (root / "CMakeLists.txt").write_text("project(x)\n")
    (root / "src" / "model.cpp").write_text("int rate() { return 1; }\n")
    for path in (root / "CMakeLists.txt", root / "src" / "model.cpp"):
        os.utime(path, (1000.0, 1000.0))
    extension = tmp_path / "site-packages" / "bngsim" / "_bngsim_core.cpython-312-darwin.so"
    extension.parent.mkdir(parents=True)
    extension.write_bytes(b"compiled from int rate() { return 1; }")
    os.utime(extension, (2000.0, 2000.0))
    return root, extension


def _rewrite(path: Path, data: bytes | None = None, *, mtime: float = 5000.0) -> None:
    """Write ``data`` over ``path`` (its own bytes by default, as a checkout does) and
    stamp it newer than the extension."""
    path.write_bytes(path.read_bytes() if data is None else data)
    os.utime(path, (mtime, mtime))


@pytest.fixture
def loaded(monkeypatch: pytest.MonkeyPatch):
    """Point the guard's view of the loaded binary at a test tree and extension."""

    def point(root: Path, extension: Path) -> None:
        monkeypatch.setattr(bp, "_core_path", lambda: extension)
        monkeypatch.setattr(bp, "_source_root", lambda: root)
        monkeypatch.setattr(bp, "_build_commit", lambda: "0123456789ab")
        monkeypatch.setattr(bp, "_head_commit", lambda _root: None)
        monkeypatch.delenv("BNGSIM_NO_BUILD_CHECK", raising=False)
        monkeypatch.delenv("BNGSIM_ALLOW_STALE_CORE", raising=False)

    return point


def test_an_identical_rewrite_is_fresh_once_a_rebuild_recorded_the_sources(
    tmp_path: Path, loaded
) -> None:
    """The issue's reproducer: same bytes, newer mtime. Fresh, and the banner says why."""
    root, extension = _checkout(tmp_path)
    loaded(root, extension)
    bp.write_source_record(extension, bp.source_digest(root))
    _rewrite(root / "src" / "model.cpp")

    prov = bp.gather()
    assert prov.mtime_stale, "fixture is not what it claims: the timestamps say stale"
    assert prov.digest_verified
    assert not prov.is_stale
    assert bp.identity_line(prov).endswith("| fresh (digest)")
    # Every reader of the verdict agrees: the preflight, is_stale(), capabilities().
    assert bp.blocking_report(prov) is None
    assert bp.is_stale() is False
    assert bp.summary() == {"commit": "0123456789ab", "stale": False}


def test_a_changed_byte_is_stale_whatever_the_record_says(tmp_path: Path, loaded) -> None:
    root, extension = _checkout(tmp_path)
    loaded(root, extension)
    bp.write_source_record(extension, bp.source_digest(root))
    source = root / "src" / "model.cpp"
    _rewrite(source, source.read_bytes() + b" ")

    prov = bp.gather()
    assert prov.is_stale and not prov.digest_verified
    report = bp.format_report(prov)
    assert "STALE" in report and "model.cpp" in report


@pytest.mark.parametrize("change", ["add", "remove", "rename", "edit CMakeLists.txt"])
def test_any_change_to_the_watched_files_defeats_the_record(
    tmp_path: Path, loaded, change: str
) -> None:
    """Not only a byte: a file added, removed or renamed moves the digest too. Each
    change is followed by an identical rewrite of another source, so the timestamps
    say stale and the record is consulted."""
    root, extension = _checkout(tmp_path)
    extra = root / "src" / "extra.hpp"
    extra.write_text("// extra\n")
    os.utime(extra, (1000.0, 1000.0))
    loaded(root, extension)
    bp.write_source_record(extension, bp.source_digest(root))

    if change == "add":
        (root / "include").mkdir()
        (root / "include" / "new.hpp").write_text("// new\n")
    elif change == "remove":
        extra.unlink()
    elif change == "rename":
        # Same length, same place in the sort order: only the name itself can move it.
        extra.rename(root / "src" / "extrb.hpp")
    else:
        (root / "CMakeLists.txt").write_text("project(y)\n")
    _rewrite(root / "src" / "model.cpp")

    prov = bp.gather()
    assert prov.is_stale and not prov.digest_verified


def test_files_the_guard_does_not_watch_leave_the_record_standing(tmp_path: Path, loaded) -> None:
    root, extension = _checkout(tmp_path)
    loaded(root, extension)
    bp.write_source_record(extension, bp.source_digest(root))
    (root / "src" / "notes.txt").write_text("not a source\n")
    (root / "python").mkdir()
    (root / "python" / "shim.cpp").write_text("// outside the watched directories\n")
    _rewrite(root / "src" / "model.cpp")

    assert not bp.gather().is_stale


def test_a_record_for_other_extension_bytes_vouches_for_nothing(tmp_path: Path, loaded) -> None:
    """A different build put in place of the recorded one (the manual cmake recipe's
    copy, say) is judged by timestamps alone, even when it keeps the old mtime."""
    root, extension = _checkout(tmp_path)
    loaded(root, extension)
    bp.write_source_record(extension, bp.source_digest(root))
    extension.write_bytes(b"compiled from something else")
    os.utime(extension, (2000.0, 2000.0))
    _rewrite(root / "src" / "model.cpp")

    prov = bp.gather()
    assert prov.is_stale and not prov.digest_verified


def test_without_a_record_the_timestamp_verdict_stands(tmp_path: Path, loaded) -> None:
    """Any binary this change never saw built: a pip install, a wheel build, an old tree."""
    root, extension = _checkout(tmp_path)
    loaded(root, extension)
    _rewrite(root / "src" / "model.cpp")

    prov = bp.gather()
    assert prov.is_stale and not prov.digest_verified


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        '{"schema": 2, "extension_sha256": "EXT", "source_digest": "SRC"}',
        '{"schema": 1, "extension_sha256": "EXT"}',
        '{"schema": 1, "extension_sha256": 7, "source_digest": "SRC"}',
    ],
    ids=["not-json", "not-an-object", "other-schema", "missing-digest", "non-string-hash"],
)
def test_a_malformed_record_is_no_record(tmp_path: Path, loaded, text: str) -> None:
    """Right hashes under the wrong schema included: only a record this guard can read
    in full vouches for anything, and none of these makes gather() raise."""
    root, extension = _checkout(tmp_path)
    loaded(root, extension)
    record = text.replace("EXT", bp._file_sha256(extension))
    record = record.replace("SRC", bp.source_digest(root) or "")
    bp.source_record_path(extension).write_text(record)
    _rewrite(root / "src" / "model.cpp")

    prov = bp.gather()
    assert prov.is_stale and not prov.digest_verified


def test_a_fresh_binary_never_reads_the_record(
    tmp_path: Path, loaded, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fast path stays a stat scan: the hashing runs only where a rebuild was
    about to be demanded."""
    root, extension = _checkout(tmp_path)
    loaded(root, extension)

    def must_not_run(*_a: object, **_k: object) -> bool:
        raise AssertionError("the record was read on a fresh timestamp verdict")

    monkeypatch.setattr(bp, "_record_vouches", must_not_run)
    prov = bp.gather()
    assert not prov.is_stale and not prov.digest_verified
    assert bp.identity_line(prov).endswith("| fresh")


def test_the_digest_can_only_clear_a_stale_timestamp_verdict() -> None:
    stale = _prov(core_mtime=0.0, newest_mtime=1e9)
    assert stale.is_stale and not replace(stale, digest_verified=True).is_stale
    fresh = _prov(core_mtime=2000.0, newest_mtime=1000.0)
    assert not fresh.is_stale and not replace(fresh, digest_verified=True).is_stale


def test_the_digest_ignores_mtimes_and_where_the_checkout_lives(tmp_path: Path) -> None:
    root, _extension = _checkout(tmp_path)
    digest = bp.source_digest(root)
    assert digest is not None and len(digest) == 64
    _rewrite(root / "src" / "model.cpp")
    assert bp.source_digest(root) == digest
    moved = root.rename(tmp_path / "elsewhere")
    assert bp.source_digest(moved) == digest


def test_the_digest_of_an_empty_tree_is_none(tmp_path: Path) -> None:
    assert bp.source_digest(tmp_path) is None


def test_the_digest_watches_exactly_what_the_mtime_scan_watches(tmp_path: Path) -> None:
    """One owner of the watched set: a file the mtime scan reports is a file whose
    bytes move the digest, and a file it ignores is not."""
    candidates = {
        "src/model.cpp": True,
        "include/bngsim/model.hpp": True,
        "third_party/nfsim/patched.cpp": True,
        "cmake/BngsimResolvePybind11.cmake": True,
        "CMakeLists.txt": True,
        "src/notes.txt": False,
        "python/bngsim/shim.cpp": False,
        "docs/example.cpp": False,
    }
    for rel in candidates:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"// {rel}\n")
        os.utime(path, (1000.0, 1000.0))
    for rel, watched in candidates.items():
        path = tmp_path / rel
        before = bp.source_digest(tmp_path)
        os.utime(path, (9000.0, 9000.0))
        scanned = bp._newest_source(tmp_path)[0] == path
        path.write_text(f"// {rel}, edited\n")
        os.utime(path, (1000.0, 1000.0))
        moved = bp.source_digest(tmp_path) != before
        assert (scanned, moved) == (watched, watched), rel


def test_a_record_is_named_after_its_extension_and_replaced_whole(tmp_path: Path) -> None:
    root, extension = _checkout(tmp_path)
    other = extension.with_name("_bngsim_core.cpython-313-darwin.so")
    assert bp.source_record_path(extension).name == extension.name + ".source-digest.json"
    assert bp.source_record_path(extension) != bp.source_record_path(other)

    assert bp.remove_source_record(extension) is False
    path = bp.write_source_record(extension, "0" * 64)
    path = bp.write_source_record(extension, bp.source_digest(root) or "")
    assert path.is_file()
    assert not list(path.parent.glob("*.tmp")), "the temporary file was left behind"
    assert bp._record_vouches(extension, root)
    assert bp.remove_source_record(extension) is True
    assert not path.exists()
