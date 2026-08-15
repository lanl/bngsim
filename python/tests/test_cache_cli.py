"""Issue #205 — the codegen artifact cache is inspectable, sweepable, and prunable.

``~/.cache/bngsim/codegen`` grew to 2.0 GB / 14,377 entries on one developer
machine with nothing to report on it and no supported way to empty it. ``bngsim.cache``
is the answer: four verbs (``info`` / ``clean`` / ``prune`` / ``clear``) as a Python
API and as ``bngsim-cache`` / ``python -m bngsim.cache``.

Two clusters of test here, and they are testing different things.

**The classifier agrees with what ``_codegen`` actually writes.** Every removal in
``bngsim.cache`` is gated on a filename classification, so the classifier is a
*duplicate* of the naming scheme in ``_codegen`` — the exact shape that goes stale
silently. So these do not assert against hand-written name literals: they drive the
real ``get_cached_so`` / ``compile_rhs`` / ``_compile_sharded`` and classify the names
those produce. If issue #363 changes the scheme to ``rhs_<key>_<hash>``, this file
goes red rather than the sweeps quietly filing every artifact under "foreign" and
declining to remove anything.

**Nothing is removed that should not be.** The blast radius of a wrong answer here is
``rm -rf`` on a user-supplied path, so the invariants — foreign entries survive every
verb, young entries survive every verb, ``--dry-run`` deletes nothing — are asserted
per verb rather than once for a representative one.

No test needs a C compiler: ``_run_compile`` is stubbed where the real naming has to
be observed, so the compile-path coverage runs everywhere rather than skipping on the
machines least likely to have a warm cache.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import bngsim
import bngsim._codegen as cg
import pytest
from bngsim import cache as ch

# ─── Fixtures ────────────────────────────────────────────────────────────────

SUFFIX = cg._shared_lib_suffix()

#: Roughly a day, so ages are unambiguous against the one-hour default floor.
DAY = 86400.0


def _dead_pid() -> int:
    """A PID that is not running, for the ``<pid>_<counter>`` token in fixture names.

    Not a literal, because a literal is a trap: the obvious ones are alive. The first
    draft of this file spelled its partials ``rhs_<hash>.99_0.c``, and PID 99 is a
    running system daemon on macOS — so ``clean``'s liveness backstop correctly held
    every fixture back and eight tests failed for a reason that had nothing to do with
    what they were testing. Probing for a free PID says what is meant.
    """
    if os.name != "posix":  # the liveness probe is POSIX-only; any value will do
        return 999_999
    for pid in range(999_999, 100_000, -1):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return pid
        except OSError:
            continue  # alive and someone else's
    raise RuntimeError("no unused PID found")  # pragma: no cover - 900k live processes


DEAD_PID = _dead_pid()


def write_entry(
    root: Path,
    name: str,
    *,
    size: int = 1024,
    age_days: float = 30.0,
    used_days: float | None = None,
) -> Path:
    """Create one cache file with a controlled size and (mtime, atime).

    ``used_days`` defaults to ``age_days`` — built and never touched since, which is
    what a real artifact on a ``noatime`` filesystem looks like.
    """
    path = root / name
    path.write_bytes(b"\0" * size)
    now = time.time()
    mtime = now - age_days * DAY
    atime = now - (age_days if used_days is None else used_days) * DAY
    os.utime(path, (atime, mtime))
    return path


def write_shard(root: Path, name: str, *, size: int = 2048, age_days: float = 30.0) -> Path:
    """Create one ``bngsim_shard_*``-style scratch directory holding an object file."""
    work = root / name
    work.mkdir()
    (work / "unit_0000.o").write_bytes(b"\0" * size)
    stamp = time.time() - age_days * DAY
    os.utime(work / "unit_0000.o", (stamp, stamp))
    os.utime(work, (stamp, stamp))
    return work


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    """An empty cache directory, wired in as the live one.

    Monkeypatching ``_codegen.CACHE_DIR`` rather than passing ``cache_dir=``
    everywhere on purpose: that attribute is the default path every caller of the
    public API takes, and a suite that only ever exercised the explicit argument
    would not notice the default going stale.
    """
    d = tmp_path / "codegen"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def _live_cache(monkeypatch: pytest.MonkeyPatch, cache: Path) -> None:
    monkeypatch.setattr(cg, "CACHE_DIR", cache)


def stub_compiler(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Make ``compile_rhs`` run end-to-end with no C compiler, recording commands.

    The stub writes whatever the command's ``-o`` names so the caller's
    ``os.replace`` / ``os.utime`` succeed, and forces the gcc/clang command shape on
    every platform so the test is not two tests.
    """
    commands: list[list[str]] = []

    def fake_run(cmd, *, cwd=None, timeout=None):
        commands.append(list(cmd))
        out = Path(cmd[cmd.index("-o") + 1])
        if cwd is not None:
            out = Path(cwd) / out
        out.write_bytes(b"\0" * 64)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(cg, "_find_c_compiler", lambda: ["cc"])
    monkeypatch.setattr(cg, "_run_compile", fake_run)
    return commands


# ─── The classifier tracks the real naming scheme ────────────────────────────


class TestNamesComeFromCodegen:
    """The taxonomy is read off ``_codegen``'s output, not off a literal in this file."""

    @pytest.mark.parametrize(
        ("model_hash", "kind"),
        [
            ("0123456789abcdef", ch.KIND_RHS),
            ("ssaprop_0123456789abcdef", ch.KIND_SSAPROP),
            ("src_0123456789abcdef", ch.KIND_SRC),
        ],
        ids=["net-or-model", "ssa-propensity", "source-hash-fallback"],
    )
    def test_an_installed_artifact_is_the_name_get_cached_so_resolves(
        self, cache: Path, model_hash: str, kind: str
    ) -> None:
        """``get_cached_so`` finding it is what proves the name is bngsim's, and only
        then does the kind assertion mean anything."""
        path = write_entry(cache, f"rhs_{model_hash}{SUFFIX}")
        assert cg.get_cached_so(model_hash) == path
        assert ch.classify(path) == kind

    def test_compile_rhs_leaves_names_that_classify_as_its_own(
        self, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The in-flight ``.c`` and temp library are partials; the installed one is not.

        Both temp names carry ``compile_rhs``'s ``<pid>_<counter>`` token, which is the
        entire basis for telling debris from a usable artifact.
        """
        commands = stub_compiler(monkeypatch)
        installed = cg.compile_rhs("int bngsim_rhs(void) { return 0; }\n", "0123456789abcdef")

        (cmd,) = commands
        temp_lib = Path(cmd[cmd.index("-o") + 1])
        temp_c = next(Path(tok) for tok in cmd if tok.endswith(".c"))
        assert temp_lib.parent == cache and temp_c.parent == cache

        assert ch.classify(temp_c, is_dir=False) == ch.KIND_PARTIAL_C
        assert ch.classify(temp_lib, is_dir=False) == ch.KIND_PARTIAL_LIB
        assert ch.classify(installed) == ch.KIND_RHS

    def test_the_sharded_compile_scratch_dir_classifies_as_a_partial(
        self, cache: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_compile_sharded`` removes its scratch dir in a ``finally``, so one on disk
        means the process was killed — which is exactly what ``clean`` collects."""
        work_dirs: list[Path] = []

        def fake_run(cmd, *, cwd=None, timeout=None):
            work_dirs.append(Path(cwd))
            (Path(cwd) / cmd[cmd.index("-o") + 1]).write_bytes(b"\0" * 16)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        monkeypatch.setattr(cg, "_run_compile", fake_run)
        cg._compile_sharded(
            "int driver(void) { return 0; }\n",
            ["int unit0(void) { return 0; }\n"],
            cache / f"out{SUFFIX}",
            "-O2",
            2,
            None,
            ["cc"],
        )
        assert work_dirs, "the sharded compile ran no commands"
        assert ch.classify(work_dirs[0], is_dir=True) == ch.KIND_SHARD


class TestClassify:
    @pytest.mark.parametrize(
        ("name", "is_dir", "kind"),
        [
            # Every platform's suffix, not just this one: a cache directory can be
            # shared over NFS between a Linux cluster and a macOS laptop.
            ("rhs_0123456789abcdef.so", False, ch.KIND_RHS),
            ("rhs_0123456789abcdef.dylib", False, ch.KIND_RHS),
            ("rhs_0123456789abcdef.dll", False, ch.KIND_RHS),
            ("rhs_ssaprop_0123456789abcdef.so", False, ch.KIND_SSAPROP),
            ("rhs_src_0123456789abcdef.so", False, ch.KIND_SRC),
            ("rhs_0123456789abcdef.4711_0.c", False, ch.KIND_PARTIAL_C),
            ("rhs_0123456789abcdef.4711_0.so", False, ch.KIND_PARTIAL_LIB),
            ("rhs_src_0123456789abcdef.4711_12.dylib", False, ch.KIND_PARTIAL_LIB),
            # The MSVC sidecars of lanl/bngsim#362, which `clean` collects while the
            # leak itself waits for a change that may invalidate the cache anyway.
            ("rhs_0123456789abcdef.4711_0.lib", False, ch.KIND_PARTIAL_LIB),
            ("rhs_0123456789abcdef.4711_0.exp", False, ch.KIND_PARTIAL_LIB),
            ("bngsim_shard_a1b2c3", True, ch.KIND_SHARD),
            # Not ours, in every way a name can fail to be ours.
            ("notes.txt", False, ch.KIND_FOREIGN),
            ("rhs_0123456789abcdef.txt", False, ch.KIND_FOREIGN),
            ("rhs_0123456789abcdef.4711_0.txt", False, ch.KIND_FOREIGN),
            ("rhs_0123456789abcdef", False, ch.KIND_FOREIGN),
            ("prefix_rhs_0123456789abcdef.so", False, ch.KIND_FOREIGN),
            ("my_project", True, ch.KIND_FOREIGN),
            ("rhs_0123456789abcdef.so", True, ch.KIND_FOREIGN),
        ],
    )
    def test_kind(self, name: str, is_dir: bool, kind: str) -> None:
        assert ch.classify(name, is_dir=is_dir) == kind

    def test_every_kind_is_either_an_artifact_or_a_partial_or_foreign(self) -> None:
        """The three sets partition :data:`KINDS`. ``clean`` sweeps one, ``prune``
        evicts another, and nothing at all touches the third — a kind that fell out of
        all three would be invisible to every verb."""
        assert ch.ARTIFACT_KINDS | ch.PARTIAL_KINDS | {ch.KIND_FOREIGN} == set(ch.KINDS)
        assert not ch.ARTIFACT_KINDS & ch.PARTIAL_KINDS


# ─── info ────────────────────────────────────────────────────────────────────


class TestInfo:
    def test_counts_sizes_and_kinds(self, cache: Path) -> None:
        write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}", size=1000)
        write_entry(cache, f"rhs_{'b' * 16}{SUFFIX}", size=2000)
        write_entry(cache, f"rhs_ssaprop_{'c' * 16}{SUFFIX}", size=500)
        write_entry(cache, f"rhs_{'d' * 16}.{DEAD_PID}_0.c", size=300)
        write_shard(cache, "bngsim_shard_zz", size=700)
        write_entry(cache, "unrelated.bin", size=11)

        info = ch.codegen_cache_info()
        assert info.exists and info.path == cache
        assert len(info.entries) == 6
        assert info.total_bytes == 1000 + 2000 + 500 + 300 + 700 + 11
        assert info.by_kind[ch.KIND_RHS] == (2, 3000)
        assert info.by_kind[ch.KIND_SSAPROP] == (1, 500)
        assert info.by_kind[ch.KIND_PARTIAL_C] == (1, 300)
        assert info.by_kind[ch.KIND_SHARD] == (1, 700)
        assert info.by_kind[ch.KIND_FOREIGN] == (1, 11)
        assert info.partial_bytes == 1000

    def test_reports_every_kind_including_the_absent_ones(self, cache: Path) -> None:
        """A zero row is information: it says the taxonomy has a slot for that kind and
        this cache has none, rather than leaving the reader to wonder."""
        write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}")
        assert set(ch.codegen_cache_info().by_kind) == set(ch.KINDS)

    def test_a_missing_cache_directory_is_empty_not_an_error(self, tmp_path: Path) -> None:
        """Nothing has been compiled yet is the normal state of a fresh install, and it
        must not be an exception in a notebook."""
        info = ch.codegen_cache_info(tmp_path / "never-created")
        assert not info.exists
        assert info.entries == () and info.total_bytes == 0
        assert info.built_span == (None, None)

    def test_spans_bracket_the_entries(self, cache: Path) -> None:
        write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}", age_days=40, used_days=3)
        write_entry(cache, f"rhs_{'b' * 16}{SUFFIX}", age_days=10, used_days=10)
        info = ch.codegen_cache_info()
        oldest_built, newest_built = info.built_span
        assert oldest_built is not None and newest_built is not None
        now = time.time()
        assert (now - oldest_built) / DAY == pytest.approx(40, abs=0.1)
        assert (now - newest_built) / DAY == pytest.approx(10, abs=0.1)
        assert (now - info.used_span[1]) / DAY == pytest.approx(3, abs=0.1)

    def test_atime_is_live_reports_whether_the_filesystem_records_access(
        self, cache: Path
    ) -> None:
        """The signal that says whether prune's LRU order is real recency or build order.

        Worth reporting rather than assuming, because it varies by mount: a plain
        ``read()`` on macOS APFS leaves ``atime`` alone while ``dlopen`` advances it,
        and a ``noatime`` mount never advances it at all.
        """
        write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}", age_days=40, used_days=40)
        assert not ch.codegen_cache_info().atime_is_live
        write_entry(cache, f"rhs_{'b' * 16}{SUFFIX}", age_days=40, used_days=2)
        assert ch.codegen_cache_info().atime_is_live

    def test_to_dict_is_json_serializable_with_stable_keys(self, cache: Path) -> None:
        write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}", size=64)
        payload = json.loads(json.dumps(ch.codegen_cache_info().to_dict()))
        assert payload["entries"] == 1
        assert payload["total_bytes"] == 64
        assert payload["codegen_key"] == cg._CODEGEN_CACHE_KEY
        assert set(payload["by_kind"]) == set(ch.KINDS)
        assert payload["by_kind"]["rhs"] == {"entries": 1, "bytes": 64}


# ─── clean ───────────────────────────────────────────────────────────────────


class TestClean:
    def test_removes_partials_and_only_partials(self, cache: Path) -> None:
        artifact = write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}")
        stray_c = write_entry(cache, f"rhs_{'b' * 16}.{DEAD_PID}_0.c")
        temp_lib = write_entry(cache, f"rhs_{'b' * 16}.{DEAD_PID}_0{SUFFIX}")
        shard = write_shard(cache, "bngsim_shard_zz")

        sweep = ch.clean_codegen_cache()

        assert {e.path for e in sweep.removed} == {stray_c, temp_lib, shard}
        assert artifact.exists(), "clean must never cost a cache hit"
        assert not stray_c.exists() and not temp_lib.exists() and not shard.exists()

    def test_holds_back_anything_inside_the_min_age_floor(self, cache: Path) -> None:
        """A compile in flight writes its scratch files into this very directory, so a
        sweep with no margin can delete the ``.c`` a running ``cc`` is reading."""
        fresh = write_entry(cache, f"rhs_{'a' * 16}.{DEAD_PID}_0.c", age_days=0)
        old = write_entry(cache, f"rhs_{'b' * 16}.{DEAD_PID}_1.c", age_days=1)

        sweep = ch.clean_codegen_cache()

        assert [e.path for e in sweep.removed] == [old]
        assert [e.path for e in sweep.held] == [fresh]
        assert fresh.exists()

    def test_min_age_zero_takes_everything(self, cache: Path) -> None:
        fresh = write_entry(cache, f"rhs_{'a' * 16}.{DEAD_PID}_0.c", age_days=0)
        assert ch.clean_codegen_cache(min_age=0).removed_bytes == 1024
        assert not fresh.exists()

    def test_dry_run_reports_without_deleting(self, cache: Path) -> None:
        stray = write_entry(cache, f"rhs_{'b' * 16}.{DEAD_PID}_0.c", size=128)
        sweep = ch.clean_codegen_cache(dry_run=True)
        assert sweep.dry_run and [e.path for e in sweep.removed] == [stray]
        assert sweep.removed_bytes == 128
        assert stray.exists()

    def test_an_empty_cache_is_a_no_op(self, cache: Path) -> None:
        sweep = ch.clean_codegen_cache()
        assert sweep.removed == () and sweep.failed == ()
        assert cache.is_dir()


# ─── prune ───────────────────────────────────────────────────────────────────


class TestPrune:
    def test_needs_a_bound(self, cache: Path) -> None:
        """Unbounded prune is either a no-op or a clear, and guessing which is not its
        business."""
        with pytest.raises(ValueError, match="needs a bound"):
            ch.prune_codegen_cache()

    def test_older_than_evicts_by_last_use_not_by_build_time(self, cache: Path) -> None:
        """The distinction the whole LRU rests on: an artifact compiled months ago and
        loaded this morning is hot, and a size- or age-bound that threw it away would
        be re-paying its compile for nothing."""
        stale = write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}", age_days=90, used_days=90)
        old_but_used = write_entry(cache, f"rhs_{'b' * 16}{SUFFIX}", age_days=90, used_days=2)

        sweep = ch.prune_codegen_cache(older_than="30d")

        assert [e.path for e in sweep.removed] == [stale]
        assert old_but_used.exists()

    @pytest.mark.parametrize("older_than", ["30d", "720h", "43200m", 30 * DAY])
    def test_equivalent_windows_select_the_same_entries(
        self, cache: Path, older_than: str | float
    ) -> None:
        write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}", age_days=90)
        write_entry(cache, f"rhs_{'b' * 16}{SUFFIX}", age_days=2)
        sweep = ch.prune_codegen_cache(older_than=older_than, dry_run=True)
        assert [e.path.name for e in sweep.removed] == [f"rhs_{'a' * 16}{SUFFIX}"]

    def test_max_size_evicts_least_recently_used_until_it_fits(self, cache: Path) -> None:
        write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}", size=400, used_days=90)
        write_entry(cache, f"rhs_{'b' * 16}{SUFFIX}", size=400, used_days=60)
        keep_1 = write_entry(cache, f"rhs_{'c' * 16}{SUFFIX}", size=400, used_days=30)
        keep_2 = write_entry(cache, f"rhs_{'d' * 16}{SUFFIX}", size=400, used_days=10)

        sweep = ch.prune_codegen_cache(max_size=900)

        assert [e.path.name for e in sweep.removed] == [
            f"rhs_{'a' * 16}{SUFFIX}",
            f"rhs_{'b' * 16}{SUFFIX}",
        ]
        assert sweep.total_bytes_after == 800 <= 900
        assert keep_1.exists() and keep_2.exists()

    def test_max_size_stops_as_soon_as_it_fits(self, cache: Path) -> None:
        """Not "evict until well under" — one byte over the cap costs one entry."""
        write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}", size=400, used_days=90)
        write_entry(cache, f"rhs_{'b' * 16}{SUFFIX}", size=400, used_days=60)
        sweep = ch.prune_codegen_cache(max_size=799, dry_run=True)
        assert len(sweep.removed) == 1

    def test_the_min_age_floor_applies_to_the_lru_pass(self, cache: Path) -> None:
        """A cache that is entirely fresh and over the cap would otherwise nominate an
        artifact compiled minutes ago — possibly the one a sibling process is about to
        ``dlopen``. The cap is a target; not breaking a live run is not."""
        fresh = write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}", size=4000, age_days=0)

        sweep = ch.prune_codegen_cache(max_size=100)

        assert sweep.removed == ()
        assert [e.path for e in sweep.held] == [fresh]
        assert sweep.total_bytes_after > 100, "over the cap, and correctly so"
        assert fresh.exists()

    def test_the_floor_is_measured_from_use_not_from_build_time(self, cache: Path) -> None:
        """The hazard is the window between ``get_cached_so`` returning a path and the
        loader opening it, so what has to be recent is the *use*.

        An artifact compiled last month and ``dlopen``ed a minute ago is exactly what an
        LRU pass nominates when everything else in the cache is newer — and an mtime
        floor would wave it through while another process was opening it.
        """
        minute = 1.0 / 1440.0
        # Built a month ago, loaded half an hour ago — and still the LRU pick, because
        # everything else in this cache was loaded more recently again.
        just_loaded = write_entry(
            cache, f"rhs_{'a' * 16}{SUFFIX}", size=4000, age_days=30, used_days=30 * minute
        )
        write_entry(cache, f"rhs_{'b' * 16}{SUFFIX}", size=4000, age_days=30, used_days=1 * minute)

        sweep = ch.prune_codegen_cache(max_size=4000)

        assert [e.path for e in sweep.held] == [just_loaded]
        assert just_loaded.exists()

    def test_subsumes_clean_so_the_cap_covers_the_whole_directory(self, cache: Path) -> None:
        """``--max-size`` is a bound on the directory, so debris that counts toward it
        has to be collectable before artifacts are thrown away to make room for it."""
        stray = write_entry(cache, f"rhs_{'z' * 16}.{DEAD_PID}_0.c", size=600)
        artifact = write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}", size=400, used_days=90)

        sweep = ch.prune_codegen_cache(max_size=500)

        assert [e.path for e in sweep.removed] == [stray]
        assert artifact.exists(), "the artifact was never the thing over the cap"
        assert sweep.total_bytes_after == 400

    def test_both_bounds_compose(self, cache: Path) -> None:
        write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}", size=400, used_days=90)
        write_entry(cache, f"rhs_{'b' * 16}{SUFFIX}", size=400, used_days=20)
        write_entry(cache, f"rhs_{'c' * 16}{SUFFIX}", size=400, used_days=10)

        sweep = ch.prune_codegen_cache(older_than="30d", max_size=500)

        assert len(sweep.removed) == 2, "one for age, one more for size"
        assert sweep.total_bytes_after == 400

    def test_an_entry_is_never_selected_twice(self, cache: Path) -> None:
        """Both bounds nominate the same stale artifact; double-counting it would make
        the projected size wrong and evict a live one to cover the phantom."""
        write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}", size=400, used_days=90)
        keep = write_entry(cache, f"rhs_{'b' * 16}{SUFFIX}", size=400, used_days=1)

        sweep = ch.prune_codegen_cache(older_than="30d", max_size=400)

        assert len(sweep.removed) == 1
        assert keep.exists()

    def test_eviction_order_is_deterministic_when_timestamps_tie(self, cache: Path) -> None:
        """Two runs over the same cache must make the same choice; a tie broken by dict
        or directory order would make prune's effect depend on the filesystem."""
        for letter in "abcd":
            write_entry(cache, f"rhs_{letter * 16}{SUFFIX}", size=400, used_days=30)
        first = ch.prune_codegen_cache(max_size=900, dry_run=True)
        second = ch.prune_codegen_cache(max_size=900, dry_run=True)
        assert [e.path for e in first.removed] == [e.path for e in second.removed]
        assert [e.path.name for e in first.removed] == [
            f"rhs_{'a' * 16}{SUFFIX}",
            f"rhs_{'b' * 16}{SUFFIX}",
        ]


# ─── clear ───────────────────────────────────────────────────────────────────


class TestClear:
    def test_removes_artifacts_and_partials_and_keeps_the_directory(self, cache: Path) -> None:
        write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}")
        write_entry(cache, f"rhs_{'b' * 16}.{DEAD_PID}_0.c")
        write_shard(cache, "bngsim_shard_zz")

        sweep = ch.clear_codegen_cache()

        assert len(sweep.removed) == 3
        assert list(cache.iterdir()) == []
        assert cache.is_dir(), "compile_rhs recreates it, but a provisioned dir is not ours"

    def test_takes_fresh_entries_too(self, cache: Path) -> None:
        """Unlike the other verbs. "Clear" is an explicit instruction to empty the
        cache, and a floor that quietly left the newest entries behind would make it a
        lie — the guard is available as ``min_age=`` for a caller who wants it."""
        fresh = write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}", age_days=0)
        assert len(ch.clear_codegen_cache().removed) == 1
        assert not fresh.exists()

        again = write_entry(cache, f"rhs_{'b' * 16}{SUFFIX}", age_days=0)
        assert ch.clear_codegen_cache(min_age=ch.DEFAULT_MIN_AGE).removed == ()
        assert again.exists()

    def test_still_holds_a_partial_whose_compile_is_running(self, cache: Path) -> None:
        """The one thing clear's zero floor does not switch off. Emptying a cache is a
        reason to reclaim disk, not a reason to break a build that is in progress."""
        if os.name != "posix":  # pragma: no cover - the probe is POSIX-only by design
            pytest.skip("POSIX-specific: os.kill(pid, 0) terminates the process on Windows")
        live = write_entry(cache, f"rhs_{'a' * 16}.{os.getpid()}_0.c", age_days=0)
        gone = write_entry(cache, f"rhs_{'b' * 16}{SUFFIX}", age_days=0)

        sweep = ch.clear_codegen_cache()

        assert [e.path for e in sweep.removed] == [gone]
        assert [e.path for e in sweep.held] == [live]
        assert live.exists()


# ─── Safety invariants, per verb ─────────────────────────────────────────────


def sweep_verbs():
    """Every mutating verb, as ``(id, callable)``. Parametrizing the invariants over
    this rather than spot-checking one of them is the point: an invariant that holds
    for ``clean`` and not for ``clear`` is the bug that matters."""
    return [
        pytest.param(lambda **kw: ch.clean_codegen_cache(**kw), id="clean"),
        pytest.param(
            lambda **kw: ch.prune_codegen_cache(older_than=0, max_size=0, **kw), id="prune"
        ),
        pytest.param(lambda **kw: ch.clear_codegen_cache(**kw), id="clear"),
    ]


@pytest.mark.parametrize("verb", sweep_verbs())
def test_no_verb_removes_a_foreign_entry(cache: Path, verb) -> None:
    """The cache directory is user-supplied (``BNGSIM_CODEGEN_CACHE_DIR``) and people
    point it at scratch space they share with other things. A sweep that deleted
    whatever it found there would be ``rm -rf`` with someone else's aim.
    """
    foreign = [
        write_entry(cache, "notes.txt", age_days=365),
        write_entry(cache, "rhs_0123456789abcdef.txt", age_days=365),
        write_entry(cache, f"prefix_rhs_0123456789abcdef{SUFFIX}", age_days=365),
        write_shard(cache, "someone_elses_dir", age_days=365),
    ]
    write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}", age_days=365)

    sweep = verb(min_age=0)

    assert all(p.exists() for p in foreign)
    assert not any(e.path in set(foreign) for e in sweep.removed)


@pytest.mark.parametrize("verb", sweep_verbs())
def test_dry_run_deletes_nothing(cache: Path, verb) -> None:
    paths = [
        write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}", age_days=365),
        write_entry(cache, f"rhs_{'b' * 16}.{DEAD_PID}_0.c", age_days=365),
        write_shard(cache, "bngsim_shard_zz", age_days=365),
    ]
    before = ch.codegen_cache_info().total_bytes

    sweep = verb(min_age=0, dry_run=True)

    assert sweep.dry_run and sweep.removed, "a dry run still has to say what it would do"
    assert all(p.exists() for p in paths)
    assert ch.codegen_cache_info().total_bytes == before


@pytest.mark.parametrize("verb", sweep_verbs())
def test_an_entry_that_vanishes_mid_sweep_is_not_a_failure(
    cache: Path, verb, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent Dask workers sweep the same cache, and ``force_recompile`` unlinks
    artifacts out from under one. Both land on a missing path, and both got what they
    wanted — reporting an error there would make a routine race look like a fault."""
    doomed = write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}", age_days=365)
    write_entry(cache, f"rhs_{'b' * 16}.{DEAD_PID}_0.c", age_days=365)

    real_scan = ch._scan

    def scan_then_delete(cache_dir):
        entries = list(real_scan(cache_dir))
        doomed.unlink()
        return iter(entries)

    monkeypatch.setattr(ch, "_scan", scan_then_delete)
    sweep = verb(min_age=0)
    assert sweep.failed == ()


def test_a_live_compile_holds_its_own_partial(cache: Path) -> None:
    """The ``min_age`` floor is calibrated against the *default* 600 s compile budget,
    but ``BNGSIM_CODEGEN_TIMEOUT=0`` on a genome-scale model is a documented setup
    where one compile runs for tens of minutes. The PID in the name is the backstop.
    """
    if os.name != "posix":  # pragma: no cover - the probe is POSIX-only by design
        pytest.skip("POSIX-specific: os.kill(pid, 0) terminates the process on Windows")
    mine = write_entry(cache, f"rhs_{'a' * 16}.{os.getpid()}_0.c", age_days=365)
    dead = write_entry(cache, f"rhs_{'b' * 16}.{DEAD_PID}_0.c", age_days=365)

    sweep = ch.clean_codegen_cache(min_age=0)

    assert [e.path for e in sweep.held] == [mine]
    assert mine.exists() and not dead.exists()


def test_a_failed_removal_is_reported_rather_than_raised(
    cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read-only or foreign-owned cache should produce a diagnosable report and a
    non-zero exit, not a traceback out of a maintenance command."""
    write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}", age_days=365)

    def refuse(self, missing_ok: bool = False):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "unlink", refuse)
    sweep = ch.clear_codegen_cache()
    assert sweep.removed == ()
    assert [p.name for p, _ in sweep.failed] == [f"rhs_{'a' * 16}{SUFFIX}"]


# ─── Value parsing ───────────────────────────────────────────────────────────


class TestParsers:
    @pytest.mark.parametrize(
        ("text", "seconds"),
        [
            ("45s", 45),
            ("90m", 5400),
            ("12h", 43200),
            ("30d", 2592000),
            ("2w", 1209600),
            ("1.5h", 5400),
            ("  30d  ", 2592000),
            ("30D", 2592000),
            # A bare number is days: the reading of `--older-than 30` a user intends.
            ("30", 2592000),
        ],
    )
    def test_duration(self, text: str, seconds: float) -> None:
        assert ch.parse_duration(text) == seconds

    def test_a_numeric_duration_is_already_seconds(self) -> None:
        """Because it comes from the API, where ``min_age=DEFAULT_MIN_AGE`` (3600.0)
        has to mean an hour and not a decade."""
        assert ch.parse_duration(3600.0) == 3600.0
        assert ch.parse_duration(ch.DEFAULT_MIN_AGE) == 3600.0

    @pytest.mark.parametrize(
        ("text", "size"),
        [
            ("1024", 1024),
            ("2G", 2 * 1024**3),
            ("2GB", 2 * 1024**3),
            ("1.5GiB", int(1.5 * 1024**3)),
            ("500M", 500 * 1024**2),
            ("512k", 512 * 1024),
            ("1T", 1024**4),
            ("  2g  ", 2 * 1024**3),
        ],
    )
    def test_size(self, text: str, size: int) -> None:
        assert ch.parse_size(text) == size

    @pytest.mark.parametrize("bad", ["", "soon", "30x", "-5d", "d30", "1..2h"])
    def test_a_bad_duration_says_what_the_forms_are(self, bad: str) -> None:
        with pytest.raises(ValueError, match="s/m/h/d/w"):
            ch.parse_duration(bad)

    @pytest.mark.parametrize("bad", ["", "big", "2Q", "-1G", "G2"])
    def test_a_bad_size_says_what_the_forms_are(self, bad: str) -> None:
        with pytest.raises(ValueError, match="K/M/G/T"):
            ch.parse_size(bad)

    @pytest.mark.parametrize("negative", [-1, -0.5])
    def test_negative_values_are_refused(self, negative: float) -> None:
        with pytest.raises(ValueError, match="negative"):
            ch.parse_duration(negative)
        with pytest.raises(ValueError, match="negative"):
            ch.parse_size(negative)


# ─── Command line ────────────────────────────────────────────────────────────


class TestCli:
    def test_info_json_matches_the_api(self, cache: Path, capsys) -> None:
        write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}", size=64)
        assert ch.main(["info", "--json"]) == 0
        assert json.loads(capsys.readouterr().out) == ch.codegen_cache_info().to_dict()

    def test_info_text_names_the_path_and_the_kinds(self, cache: Path, capsys) -> None:
        write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}", size=64)
        assert ch.main(["info"]) == 0
        out = capsys.readouterr().out
        assert str(cache) in out
        assert cg._CODEGEN_CACHE_KEY in out
        assert "rhs" in out

    @pytest.mark.parametrize("position", ["before", "after"], ids=["-C info", "info -C"])
    def test_cache_dir_flag_overrides_the_configured_cache(
        self, tmp_path: Path, capsys, position: str
    ) -> None:
        """So a login node can audit the pre-warmed artifact directory a cluster job
        will read, without re-``exec``ing under a different environment.

        Accepted on either side of the verb: argparse gives you options-before-verb for
        free, and ``bngsim-cache info -C /scratch`` is what a user types. The post-verb
        copy is ``SUPPRESS``-defaulted so it cannot overwrite a pre-verb one with
        ``None`` — which is what a plain default there would do.
        """
        other = tmp_path / "elsewhere"
        other.mkdir()
        write_entry(other, f"rhs_{'a' * 16}{SUFFIX}", size=77)
        argv = (
            ["-C", str(other), "info", "--json"]
            if position == "before"
            else ["info", "--json", "-C", str(other)]
        )
        assert ch.main(argv) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["path"] == str(other) and payload["total_bytes"] == 77

    def test_a_pre_verb_cache_dir_is_honored_by_a_mutating_verb(
        self, cache: Path, tmp_path: Path, capsys
    ) -> None:
        """The argparse trap this guards is not cosmetic: a subparser option with a real
        default is applied *after* the top-level parse, so ``-C`` before the verb would
        be replaced by ``None`` and ``clear`` would empty the configured cache instead
        of the one that was named. Wrong directory, and no error to say so."""
        other = tmp_path / "elsewhere"
        other.mkdir()
        theirs = write_entry(other, f"rhs_{'a' * 16}{SUFFIX}", age_days=365)
        ours = write_entry(cache, f"rhs_{'b' * 16}{SUFFIX}", age_days=365)

        assert ch.main(["-C", str(other), "clear", "--yes"]) == 0
        capsys.readouterr()

        assert not theirs.exists(), "the named cache is the one that was cleared"
        assert ours.exists(), "the configured cache was not touched"

    def test_prune_without_a_bound_is_a_usage_error(self, cache: Path, capsys) -> None:
        assert ch.main(["prune"]) == 2
        assert "--older-than" in capsys.readouterr().err

    def test_prune_says_when_it_could_not_reach_the_cap(self, cache: Path, capsys) -> None:
        """Silence here would read as success, and the user would come back to a cache
        still over its cap with no idea why."""
        write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}", size=4000, age_days=0)
        assert ch.main(["prune", "--max-size", "100"]) == 0
        assert "still over" in capsys.readouterr().out

    @pytest.mark.parametrize("flag", ["-n", "--dry-run"])
    def test_dry_run_says_would(self, cache: Path, capsys, flag: str) -> None:
        write_entry(cache, f"rhs_{'a' * 16}.{DEAD_PID}_0.c", age_days=365)
        assert ch.main(["clean", flag]) == 0
        assert "would remove 1" in capsys.readouterr().out

    def test_clear_refuses_without_yes_when_stdin_is_not_a_terminal(
        self, cache: Path, capsys
    ) -> None:
        """A cron job or a CI step that reaches ``clear`` by accident should fail
        visibly, not silently throw away an afternoon of compiles."""
        kept = write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}", age_days=365)
        assert ch.main(["clear"]) == 1
        assert "--yes" in capsys.readouterr().err
        assert kept.exists()

    def test_clear_with_yes_proceeds(self, cache: Path, capsys) -> None:
        gone = write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}", age_days=365)
        assert ch.main(["clear", "--yes"]) == 0
        assert not gone.exists()
        assert "removed 1" in capsys.readouterr().out

    def test_clear_dry_run_needs_no_confirmation(self, cache: Path, capsys) -> None:
        kept = write_entry(cache, f"rhs_{'a' * 16}{SUFFIX}", age_days=365)
        assert ch.main(["clear", "--dry-run"]) == 0
        assert kept.exists()

    def test_a_failed_removal_exits_non_zero(
        self, cache: Path, capsys, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_entry(cache, f"rhs_{'a' * 16}.{DEAD_PID}_0.c", age_days=365)

        def refuse(self, missing_ok: bool = False):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(Path, "unlink", refuse)
        assert ch.main(["clean"]) == 1
        assert "error:" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "bad", [["prune", "--older-than", "soon"], ["prune", "--max-size", "?"]]
    )
    def test_an_unparsable_value_is_a_usage_error_not_a_traceback(self, bad: list[str]) -> None:
        with pytest.raises(SystemExit) as exc:
            ch.main(bad)
        assert exc.value.code == 2


class TestModuleEntryPoint:
    def test_python_dash_m_runs_and_does_not_warn(self, tmp_path: Path) -> None:
        """``python -m bngsim.cache`` is the form that works without reinstalling, which
        matters when the point is to reach a machine whose cache is already 2 GB.

        It also pins the package layout. ``bngsim/__init__.py`` imports ``bngsim.cache``,
        so a single-file ``cache.py`` would make ``runpy`` re-execute an already-imported
        module and print a warning above every report; the ``__main__.py`` shim is what
        avoids it, exactly as ``bngsim.convert`` does.
        """
        result = subprocess.run(
            [sys.executable, "-m", "bngsim.cache", "-C", str(tmp_path), "info"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert str(tmp_path) in result.stdout
        assert "found in sys.modules" not in result.stderr


def test_the_four_verbs_are_on_the_top_level_namespace() -> None:
    """A fitting harness bounding its own cache should not have to know that the
    implementation lives in a submodule."""
    for name in (
        "codegen_cache_info",
        "clean_codegen_cache",
        "prune_codegen_cache",
        "clear_codegen_cache",
    ):
        assert name in bngsim.__all__
        assert getattr(bngsim, name) is getattr(ch, name)
