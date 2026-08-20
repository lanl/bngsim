"""bngsim.cache — inspect, sweep, and prune the codegen artifact cache (issue #205).

``~/.cache/bngsim/codegen`` (or wherever ``BNGSIM_CODEGEN_CACHE_DIR`` points) fills
with compiled ``.so``/``.dylib``/``.dll`` RHS artifacts and never empties. Six weeks
of ordinary work on one developer machine left 2.0 GB across 14,377 entries, and the
only remedy was ``rm -rf`` on a path you had to know by heart.

It fills faster than "a cache grows" suggests, because the key is deliberately
conservative: :data:`bngsim._codegen._CODEGEN_CACHE_KEY` folds a digest of the
emitters' own source (issue #51), so *any* edit to ``_codegen.py`` / ``_jacobian.py``
/ ``_saturable_jacobian.py`` / ``_switch_sensitivity.py`` — a comment included —
orphans every artifact on the machine at once. That is the right trade (the
alternative, under-invalidation, is a silently wrong gradient), but it means the
directory accumulates a full corpus of dead artifacts per emitter edit.

Which is why an artifact's name carries the key that built it — ``rhs_<key>_<hash>``
(issue #363). The key is mixed into ``<hash>`` as well, so it is what decides
validity; carrying it *beside* the hash is what makes the dead corpus **countable**.
:func:`codegen_cache_info` reports live and orphaned as two numbers, and
``prune --orphaned`` sweeps exactly the artifacts this install can never load again —
much better targeted than an age bound, which throws away live artifacts that merely
have not been used lately.

Four verbs, also on the command line as ``bngsim-cache`` / ``python -m bngsim.cache``:

* :func:`codegen_cache_info` — what is in there, by kind and by key, with sizes and dates.
* :func:`clean_codegen_cache` — leaked partials only. No artifact is touched.
* :func:`prune_codegen_cache` — an orphan, age and/or size bound; LRU within it.
* :func:`clear_codegen_cache` — everything bngsim owns.

Three properties hold across all of them.

**Only bngsim's own files are ever removed.** ``BNGSIM_CODEGEN_CACHE_DIR`` is a
user-supplied path and people point it at scratch directories they share with other
things; a sweep that deleted whatever it found there would be a footgun with the
blast radius of ``rm -rf $BNGSIM_CODEGEN_CACHE_DIR``. Every entry is classified by
name first (see :func:`classify`), and anything unrecognized is counted as
:data:`KIND_FOREIGN`, reported, and left alone — by ``clear`` as much as by ``clean``.

**Nothing young is removed.** Every mutating verb holds off on anything used or
written within ``min_age`` (default one hour, comfortably over the 600 s default
``BNGSIM_CODEGEN_TIMEOUT``), and on POSIX additionally holds a partial whose owning
process is still alive. A compile in flight writes its ``.c`` and its shard directory
*into this directory*, so a sweep with no margin can break a build that is running.

**Nothing prunes automatically.** There is no size cap enforced at ``compile_rhs``
time and no sweep on import or on ``Simulator`` construction. A library that deletes
files as a side effect of being used is a surprise, and the failure mode — evicting
the artifact another process is about to ``dlopen`` — is the exact class of bug the
cache exists to avoid. Pruning happens when a human or a cron job asks for it.

Deleting an artifact that some process has already ``dlopen``ed is safe on POSIX:
the inode outlives the unlink and the mapping keeps working. The window that is *not*
safe is between :func:`bngsim._codegen.get_cached_so` returning a path and the loader
opening it, which is what the ``min_age`` floor exists to keep clear of.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import os
import re
import shutil
import sys
import textwrap
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePath

from bngsim import _codegen

__all__ = [
    "ARTIFACT_KINDS",
    "DEFAULT_MIN_AGE",
    "KINDS",
    "KIND_FOREIGN",
    "KIND_NOTE",
    "KIND_PARTIAL_C",
    "KIND_PARTIAL_LIB",
    "KIND_RHS",
    "KIND_SHARD",
    "KIND_SRC",
    "KIND_SSAPROP",
    "PARTIAL_KINDS",
    "CacheEntry",
    "CacheInfo",
    "CacheSweep",
    "artifact_key",
    "classify",
    "clean_codegen_cache",
    "clear_codegen_cache",
    "codegen_cache_dir",
    "codegen_cache_info",
    "main",
    "parse_duration",
    "parse_size",
    "prune_codegen_cache",
]


# ─── Entry kinds ─────────────────────────────────────────────────────────────
#
# The taxonomy is the artifact-naming scheme in _codegen, read backwards. Every
# compiled artifact is `rhs_{key}_{model_hash}{suffix}` (_artifact_stem); three
# model_hash namespaces are in use, and they are worth separating in a report
# because they answer different questions about where the bloat came from.
KIND_RHS = "rhs"
"""A compiled model RHS (with whatever Jacobian / output / sensitivity callbacks
the run asked for appended). The bulk of any real cache."""

KIND_SSAPROP = "ssaprop"
"""A compiled SSA propensity vector — ``rhs_<key>_ssaprop_<hash>``, its own
namespace so it can never collide with an RHS artifact."""

KIND_SRC = "src"
"""``rhs_<key>_src_<hash>``: the source-hash fallback key (issue #174), used when the
structural model key cannot be serialized. Its presence in a cache is a signal —
it means some model took the slow path that re-derives the source every time."""

KIND_PARTIAL_C = "partial-c"
"""A ``rhs_<key>_<hash>.<pid>_<n>.c`` left behind by an interrupted compile.
``compile_rhs`` unlinks it in a ``finally``, so one on disk means the process died
outright."""

KIND_PARTIAL_LIB = "partial-lib"
"""The library half of the same leak: the process-unique temp artifact a killed
compile never got to ``os.replace()`` into place, plus the ``.lib``/``.exp``
sidecars MSVC writes beside its ``/Fe:`` output."""

KIND_SHARD = "shard"
"""A ``bngsim_shard_*`` scratch directory from a parallel (sharded) compile,
holding ``.o`` files. Removed in a ``finally``, so one on disk means SIGKILL."""

KIND_NOTE = "note"
"""A ``rhs_<key>_<hash>.sens.json`` recording why the artifact of the same name was
built without an analytic sensitivity RHS (issue #438).

A few hundred bytes, written only for a model that declined, and read back on a
cache hit so the reason survives a warm cache — which is the whole point of it,
since a warm cache generates no source and therefore derives no reason. Not an
artifact: nothing loads it, and it is worth nothing without the artifact it
describes, so ``prune`` keeps a note exactly as long as it keeps that artifact."""

KIND_FOREIGN = "foreign"
"""Anything bngsim does not recognize as its own. Counted and reported, never
removed — see the module docstring."""

KINDS: tuple[str, ...] = (
    KIND_RHS,
    KIND_SSAPROP,
    KIND_SRC,
    KIND_NOTE,
    KIND_PARTIAL_C,
    KIND_PARTIAL_LIB,
    KIND_SHARD,
    KIND_FOREIGN,
)

ARTIFACT_KINDS = frozenset({KIND_RHS, KIND_SSAPROP, KIND_SRC})
"""Kinds that are a usable compiled artifact — what ``prune`` evicts."""

PARTIAL_KINDS = frozenset({KIND_PARTIAL_C, KIND_PARTIAL_LIB, KIND_SHARD})
"""Kinds that are debris from a compile that did not finish — what ``clean`` sweeps."""

#: Human-readable labels for the report table.
_KIND_LABELS: dict[str, str] = {
    KIND_RHS: "rhs",
    KIND_SSAPROP: "ssaprop",
    KIND_SRC: "src (fallback)",
    KIND_NOTE: "decline note",
    KIND_PARTIAL_C: "partial (.c)",
    KIND_PARTIAL_LIB: "partial (lib)",
    KIND_SHARD: "shard dir",
    KIND_FOREIGN: "foreign",
}

#: Every shared-library extension ``_shared_lib_suffix`` can produce, not just this
#: platform's: a cache directory can be shared over NFS between a Linux cluster and
#: a macOS laptop, and ``info`` on either should describe the whole thing rather
#: than filing the other platform's artifacts under "foreign".
_LIB_SUFFIXES = frozenset({".so", ".dylib", ".dll"})

#: MSVC's ``cl /LD /Fe:<out>`` writes an import library and an export file beside
#: the output. ``compile_rhs`` now unlinks them (lanl/bngsim#362), but every pair
#: leaked before that fix is still on disk, and an interrupted compile still leaves
#: one — so they stay classified as partials and ``clean`` still sweeps them.
_SIDECAR_SUFFIXES = frozenset({".lib", ".exp", ".pdb", ".ilk", ".obj"})

#: The process-unique token ``compile_rhs`` appends to in-flight names:
#: ``f"{os.getpid()}_{next(_compile_counter)}"``. Its presence is what separates a
#: temp file from an installed artifact — the installed name has no such component.
_TEMP_TOKEN = re.compile(r"\.(\d+)_(\d+)$")

_SHARD_PREFIX = "bngsim_shard_"
_ARTIFACT_PREFIX = "rhs_"

#: What tells a key field from the first underscore-separated piece of a pre-#363
#: name. Read from ``_codegen`` rather than spelled here, because it is that
#: module's shape contract for the field: :func:`bngsim._codegen._artifact_key_field`
#: guarantees every key it renders carries this.
#:
#: An underscore alone cannot do the job. ``compile_rhs`` keys on whatever string
#: its caller hands it, and some contain underscores — the pre-#363
#: ``rhs_test_sens_0cec059f`` on a developer machine split into key ``test`` and
#: hash ``sens_0cec059f``, inventing a key that never existed and filing four real
#: artifacts under it in ``info``. The two namespaces (``rhs_ssaprop_<hash>``,
#: ``rhs_src_<hash>``) were the same bug with a known spelling; the marker covers
#: them and every other hash, so they need no special case.
_KEY_FIELD_MARKER = _codegen._KEY_FIELD_MARKER

#: What a decline note is named, likewise read from the module that writes them
#: rather than spelled again here: a file bngsim writes and this module does not
#: recognize would be reported as somebody else's and never swept.
_NOTE_EXT = _codegen._SENS_DECLINE_NOTE_EXT

DEFAULT_MIN_AGE = 3600.0
"""Seconds an entry must be untouched before any sweep will remove it.

One hour, chosen against the compile budget rather than picked round: the default
``BNGSIM_CODEGEN_TIMEOUT`` is 600 s, so under default settings nothing an hour old
can still belong to a live compile. Raise it (``--min-age``) on a box that builds
genome-scale models with the timeout lifted, where a single compile legitimately
runs for tens of minutes.
"""


def _split_stem(stem: str) -> tuple[str | None, str]:
    """``(codegen key, remainder)`` for a ``rhs_``-prefixed stem.

    The key is ``None`` for a name written before issue #363 put it there — a
    ``rhs_<hash>`` whose hash this install will never look up again, so it counts as
    orphaned. Telling the two apart is :data:`_KEY_FIELD_MARKER`'s job, because a
    hash can itself contain underscores (``rhs_test_sens_0cec059f``, the ``ssaprop_``
    and ``src_`` namespaces) and splitting on the first one would invent a key out of
    the front of a keyless name.

    The ``<pid>_<counter>`` token of an in-flight compile is stripped first, for the
    same reason in the other direction: without that, the legacy partial
    ``rhs_<hash>.4711_0.c`` would offer its own PID token as the split point.
    """
    m = _TEMP_TOKEN.search(stem)
    body = (stem[: m.start()] if m else stem)[len(_ARTIFACT_PREFIX) :]
    head, sep, rest = body.partition("_")
    if not sep or _KEY_FIELD_MARKER not in head:
        return None, body
    return head, rest


def _note_family(name: str) -> str:
    """The base name a decline note and the artifact it describes share.

    ``rhs_<key>_<hash>.so`` and ``rhs_<key>_<hash>.sens.json`` both reduce to
    ``rhs_<key>_<hash>``, which is how :func:`prune_codegen_cache` pairs them. A
    note still under its process-unique temporary name keeps that token and so
    matches no artifact, which is how one left behind by a killed process is
    collected rather than kept forever.
    """
    if name.endswith(_NOTE_EXT):
        return name[: -len(_NOTE_EXT)]
    suffix = PurePath(name).suffix
    return name[: -len(suffix)] if suffix else name


def classify(path: str | os.PathLike[str], *, is_dir: bool | None = None) -> str:
    """Return the :data:`KINDS` entry ``path``'s *name* identifies it as.

    Name-only, deliberately: the classifier is the safety boundary for every
    removal in this module, so it must not depend on reading, opening, or trusting
    the content of a file that some other process may be writing. ``is_dir``
    defaults to a ``stat`` of the path; pass it to classify a name in the abstract.

    Pre-#363 names (no key field) classify exactly as they always did: they are
    still bngsim's files, so ``clear`` and ``prune`` must still be able to remove
    them. What they no longer have is a key — see :func:`artifact_key`.
    """
    name = PurePath(path).name
    if is_dir is None:
        is_dir = Path(path).is_dir()
    if is_dir:
        return KIND_SHARD if name.startswith(_SHARD_PREFIX) else KIND_FOREIGN
    if not name.startswith(_ARTIFACT_PREFIX):
        return KIND_FOREIGN
    if name.endswith(_NOTE_EXT):
        # Before the temp-token branch, so the process-unique name a note is
        # written under on its way into place is still recognized as a note. It has
        # no artifact of its own name, which is what makes ``prune`` collect it.
        return KIND_NOTE
    stem, suffix = PurePath(name).stem, PurePath(name).suffix.lower()
    if _TEMP_TOKEN.search(stem):
        if suffix == ".c":
            return KIND_PARTIAL_C
        if suffix in _LIB_SUFFIXES or suffix in _SIDECAR_SUFFIXES:
            return KIND_PARTIAL_LIB
        # `rhs_x.1_2.whatever` — shaped like ours but not a name we write.
        return KIND_FOREIGN
    if suffix not in _LIB_SUFFIXES:
        return KIND_FOREIGN
    _, body = _split_stem(stem)
    if body.startswith("ssaprop_"):
        return KIND_SSAPROP
    if body.startswith("src_"):
        return KIND_SRC
    return KIND_RHS


def artifact_key(path: str | os.PathLike[str], *, is_dir: bool | None = None) -> str | None:
    """Return the codegen key ``path``'s *name* carries, or ``None`` if it carries none.

    The counterpart to :func:`classify`, and name-only for the same reason. An
    artifact is ``rhs_<key>_<hash><suffix>`` (issue #363), where ``<key>`` is
    :data:`bngsim._codegen._CODEGEN_CACHE_KEY` — the ``_CODEGEN_VERSION`` and a
    digest of the emitters' own source. Comparing it against this install's is the
    whole of "is this entry live or orphaned":

    >>> import bngsim._codegen as _cg
    >>> from bngsim.cache import artifact_key
    >>> artifact_key(f"{_cg._artifact_stem('0123456789abcdef')}.so") == _cg._CODEGEN_CACHE_KEY
    True

    ``None`` means the name carries no key field: one written before #363, or one
    bngsim does not write at all (a shard directory, anything without the ``rhs_``
    prefix). Which of those it is is :func:`classify`'s answer, not this one's —
    and the accounting only ever asks this about entries ``classify`` has already
    called an artifact.
    """
    name = PurePath(path).name
    if is_dir is None:
        is_dir = Path(path).is_dir()
    if is_dir or not name.startswith(_ARTIFACT_PREFIX):
        return None
    return _split_stem(PurePath(name).stem)[0]


@dataclass(frozen=True)
class CacheEntry:
    """One top-level entry in the cache directory — a file, or a shard directory."""

    path: Path
    kind: str
    size: int
    """Bytes on disk. For a shard directory, the sum over its whole tree."""
    mtime: float
    """When the entry was last written — for an artifact, when it was compiled."""
    last_used: float
    """``max(atime, mtime)``, the LRU key. See :func:`_stamp` for why both."""
    codegen_key: str | None = None
    """The codegen key the name carries (issue #363), or ``None`` for a foreign
    entry, a shard directory, or an artifact written under the pre-#363 scheme.
    See :func:`artifact_key`; :meth:`CacheInfo.orphaned` is what compares it."""

    @property
    def is_artifact(self) -> bool:
        return self.kind in ARTIFACT_KINDS

    @property
    def is_partial(self) -> bool:
        return self.kind in PARTIAL_KINDS

    @property
    def is_removable(self) -> bool:
        """Whether any verb in this module may delete it — i.e. bngsim wrote it."""
        return self.kind != KIND_FOREIGN


@dataclass(frozen=True)
class CacheInfo:
    """A structured census of the cache directory, as returned by
    :func:`codegen_cache_info`."""

    path: Path
    exists: bool
    entries: tuple[CacheEntry, ...]
    codegen_key: str
    """This install's :data:`bngsim._codegen._CODEGEN_CACHE_KEY` — and, since issue
    #363, the key an artifact this install builds carries in its *name*.

    So it is a filter, not just context: every entry whose
    :attr:`CacheEntry.codegen_key` differs is an artifact this install can never
    load again. :attr:`live` / :attr:`orphaned` / :attr:`by_key` are that
    comparison, and ``prune(orphaned=True)`` is the sweep it enables.
    """

    @property
    def _key_field(self) -> str:
        """:attr:`codegen_key` as a filename carries it — what entry keys are
        compared against. Identity for the shipped key format (pinned by
        ``test_codegen_cache_key.py``); the indirection exists so that a key format
        needing escaping could not silently make every entry look orphaned."""
        return _codegen._artifact_key_field(self.codegen_key)

    @property
    def total_bytes(self) -> int:
        return sum(e.size for e in self.entries)

    @property
    def live(self) -> tuple[CacheEntry, ...]:
        """Artifacts built under this install's key — the ones a run can still hit."""
        field = self._key_field
        return tuple(e for e in self.entries if e.is_artifact and e.codegen_key == field)

    @property
    def orphaned(self) -> tuple[CacheEntry, ...]:
        """Artifacts built under any *other* key, plus every pre-#363 name.

        Dead weight for this install by construction: the key is part of the content
        hash, so nothing here will ever be looked up again. On a machine that tracks
        bngsim development this is usually most of the cache — one emitter edit
        orphans every artifact at once (issue #51).

        Not necessarily dead for the *machine*: a second venv with a different bngsim
        owns its own key, and from its point of view this install's artifacts are the
        orphans. That is what ``keep_keys`` on :func:`prune_codegen_cache` is for.
        """
        field = self._key_field
        return tuple(e for e in self.entries if e.is_artifact and e.codegen_key != field)

    @property
    def live_bytes(self) -> int:
        return sum(e.size for e in self.live)

    @property
    def orphaned_bytes(self) -> int:
        return sum(e.size for e in self.orphaned)

    @property
    def by_key(self) -> dict[str | None, tuple[int, int]]:
        """``{codegen key: (entries, bytes)}`` over compiled artifacts.

        Artifacts only: a leaked partial is ``clean``'s business whatever key it
        carries, and a foreign entry is nobody's. ``None`` is the pre-#363 bucket.

        This is the audit of a shared or pre-warmed artifact directory the issue asks
        for — which bngsim versions' artifacts are in it, and how much each holds.
        """
        out: dict[str | None, tuple[int, int]] = {}
        for e in self.entries:
            if not e.is_artifact:
                continue
            n, b = out.get(e.codegen_key, (0, 0))
            out[e.codegen_key] = (n + 1, b + e.size)
        return out

    @property
    def by_kind(self) -> dict[str, tuple[int, int]]:
        """``{kind: (entries, bytes)}`` for every kind in :data:`KINDS`, zeros included."""
        out = {k: (0, 0) for k in KINDS}
        for e in self.entries:
            n, b = out.get(e.kind, (0, 0))
            out[e.kind] = (n + 1, b + e.size)
        return out

    @property
    def partial_bytes(self) -> int:
        return sum(e.size for e in self.entries if e.is_partial)

    def _span(self, attr: str) -> tuple[float | None, float | None]:
        stamps = [getattr(e, attr) for e in self.entries]
        return (min(stamps), max(stamps)) if stamps else (None, None)

    @property
    def built_span(self) -> tuple[float | None, float | None]:
        """(oldest, newest) build time over all entries, or ``(None, None)`` if empty."""
        return self._span("mtime")

    @property
    def used_span(self) -> tuple[float | None, float | None]:
        """(oldest, newest) last-used time. Equal to :attr:`built_span` entry-for-entry
        on a filesystem that does not record access times — see :attr:`atime_is_live`."""
        return self._span("last_used")

    @property
    def atime_is_live(self) -> bool:
        """Whether *any* entry has been used since it was built.

        ``False`` means either a genuinely cold cache or a filesystem that does not
        move ``atime`` (a ``noatime`` mount), and there is no way to tell which from
        here. Either way it says the same operational thing: ``prune``'s LRU order
        on this box degrades to build order.
        """
        return any(e.last_used > e.mtime + 1.0 for e in self.entries)

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable form, as ``bngsim-cache info --json`` prints it."""

        def _iso(t: float | None) -> str | None:
            return None if t is None else datetime.fromtimestamp(t).isoformat(timespec="seconds")

        built, used = self.built_span, self.used_span
        field = self._key_field
        return {
            "path": str(self.path),
            "exists": self.exists,
            "codegen_key": self.codegen_key,
            "entries": len(self.entries),
            "total_bytes": self.total_bytes,
            "live_entries": len(self.live),
            "live_bytes": self.live_bytes,
            "orphaned_entries": len(self.orphaned),
            "orphaned_bytes": self.orphaned_bytes,
            # A list, not an object: the pre-#363 bucket's key is null, which JSON
            # cannot spell as a member name. Ordered largest-first, which is the
            # order a reader deciding what to sweep wants them in.
            "by_key": [
                {"key": k, "entries": n, "bytes": b, "live": k == field}
                for k, (n, b) in sorted(
                    self.by_key.items(), key=lambda kv: (-kv[1][1], kv[0] or "")
                )
            ],
            "built_oldest": _iso(built[0]),
            "built_newest": _iso(built[1]),
            "last_used_oldest": _iso(used[0]),
            "last_used_newest": _iso(used[1]),
            "atime_is_live": self.atime_is_live,
            "by_kind": {
                k: {"entries": n, "bytes": b} for k, (n, b) in sorted(self.by_kind.items())
            },
        }


@dataclass(frozen=True)
class CacheSweep:
    """What a mutating verb did (or, under ``dry_run``, would have done)."""

    dry_run: bool
    removed: tuple[CacheEntry, ...]
    held: tuple[CacheEntry, ...]
    """Entries a verb selected but declined to remove because they are younger than
    ``min_age`` or their compile is still running. Reported rather than silently
    dropped: "clean removed nothing" and "clean found three partials and judged all
    three to be live builds" are different answers."""
    failed: tuple[tuple[Path, str], ...]
    """``(path, message)`` for entries that could not be removed (permissions, a
    read-only cache). A vanished entry is not a failure — a concurrent sweep or a
    ``force_recompile`` getting there first is the expected race, not an error."""
    total_bytes_before: int

    @property
    def removed_bytes(self) -> int:
        return sum(e.size for e in self.removed)

    @property
    def total_bytes_after(self) -> int:
        return self.total_bytes_before - self.removed_bytes


# ─── Scanning ────────────────────────────────────────────────────────────────


def codegen_cache_dir(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the cache directory: ``cache_dir`` if given, else the live one.

    Read from :data:`bngsim._codegen.CACHE_DIR` on every call rather than bound at
    import, because that attribute is the documented handle for relocating the
    cache in-process — the test suite monkeypatches it, and so do harnesses that
    want a scratch cache without re-``exec``ing under a different environment.
    """
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    return Path(_codegen.CACHE_DIR)


def _tree_size_and_mtime(root: Path) -> tuple[int, float]:
    """Total bytes and newest mtime over a shard directory's tree.

    The newest mtime, not the directory's own: a sharded compile writes ``.o`` files
    as they finish, so the freshest thing inside is the best available evidence that
    a build is still working in there. Under-counting that would let ``clean`` delete
    the scratch space of a compile that is merely slow.
    """
    total = 0
    newest = 0.0
    with contextlib.suppress(OSError):
        newest = root.stat().st_mtime
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            try:
                st = os.stat(os.path.join(dirpath, name))
            except OSError:
                continue
            total += st.st_size
            newest = max(newest, st.st_mtime)
    return total, newest


def _stamp(st: os.stat_result) -> tuple[float, float]:
    """``(mtime, last_used)`` for a stat result, where ``last_used = max(atime, mtime)``.

    Neither timestamp alone is trustworthy, and taking the newer of the two is
    correct under every combination we measured:

    * A ``noatime`` mount, and macOS APFS for ordinary ``read()``, never move
      ``atime`` — there, ``atime`` can be *older* than ``mtime`` and LRU on it alone
      would evict the artifact most recently compiled. ``mtime`` rescues that.
    * ``dlopen`` does move it, including on APFS (measured: a plain ``read()`` of a
      cached ``.dylib`` left ``atime`` untouched, ``ctypes.CDLL`` of the same file
      advanced it), and Linux's usual ``relatime`` gives day-granularity updates.
      Loading an artifact is the *only* way this cache is used, so where ``atime``
      works at all it works for exactly the access that matters.

    So the LRU order is true recency wherever the filesystem records it and degrades
    to build order — a FIFO, still bounded, never wrong — where it does not.
    """
    return st.st_mtime, max(st.st_atime, st.st_mtime)


def _scan(cache_dir: Path) -> Iterator[CacheEntry]:
    """Yield one :class:`CacheEntry` per top-level entry, skipping what vanishes.

    Entries can disappear mid-scan: a concurrent ``compile_rhs`` unlinks its temp
    ``.c`` and ``os.replace``s its artifact, and another sweep may be running. Every
    such race resolves the same way — the entry is not there, so it is not reported.
    """
    try:
        scan = os.scandir(cache_dir)
    except OSError:
        return
    with scan:
        for de in scan:
            try:
                is_dir = de.is_dir(follow_symlinks=False)
                kind = classify(de.name, is_dir=is_dir)
                if is_dir and kind == KIND_SHARD:
                    size, mtime = _tree_size_and_mtime(Path(de.path))
                    last_used = mtime
                else:
                    st = de.stat(follow_symlinks=False)
                    size = st.st_size
                    mtime, last_used = _stamp(st)
            except OSError:
                continue
            yield CacheEntry(
                path=Path(de.path),
                kind=kind,
                size=size,
                mtime=mtime,
                last_used=last_used,
                codegen_key=artifact_key(de.name, is_dir=is_dir),
            )


def codegen_cache_info(cache_dir: str | os.PathLike[str] | None = None) -> CacheInfo:
    """Census the codegen artifact cache: what is in it, by kind and by codegen key.

    Reads only directory metadata (one ``scandir``, plus a walk of any shard
    directory), so it is cheap even on the 14,000-entry cache issue #205 reports and
    never opens an artifact — the key comes off the *name* (issue #363).

    Examples
    --------
    >>> import bngsim
    >>> info = bngsim.codegen_cache_info()
    >>> info.total_bytes >= 0
    True
    >>> set(info.by_kind) == set(bngsim.cache.KINDS)
    True
    >>> len(info.live) + len(info.orphaned) == sum(
    ...     1 for e in info.entries if e.is_artifact
    ... )
    True
    """
    path = codegen_cache_dir(cache_dir)
    entries = tuple(sorted(_scan(path), key=lambda e: str(e.path)))
    return CacheInfo(
        path=path,
        exists=path.is_dir(),
        entries=entries,
        codegen_key=_codegen._CODEGEN_CACHE_KEY,
    )


# ─── Removal ─────────────────────────────────────────────────────────────────


def _compile_may_be_running(entry: CacheEntry) -> bool:
    """Whether ``entry`` looks like it belongs to a compile that is still alive.

    A partial's name carries the PID that wrote it (``rhs_<key>_<hash>.<pid>_<n>.c``),
    so
    the ``min_age`` floor can be backed up with a direct liveness check — which
    matters because the floor is calibrated against the *default* 600 s compile
    budget, and ``BNGSIM_CODEGEN_TIMEOUT=0`` with a genome-scale model is a
    documented configuration where one compile legitimately runs for tens of minutes.

    POSIX only, and that restriction is not incidental: on Windows ``os.kill(pid, 0)``
    does not probe, it calls ``TerminateProcess`` — so the "is it alive?" question
    would answer itself by killing the process. Windows relies on ``min_age`` alone.

    A recycled PID can hold a genuinely dead partial indefinitely. That is the
    conservative direction (a file we decline to delete, which ``clear`` still gets),
    so it is left as is.
    """
    if os.name != "posix":
        return False
    m = _TEMP_TOKEN.search(PurePath(entry.path).stem)
    if m is None:
        return False
    try:
        os.kill(int(m.group(1)), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Alive, owned by another user. Somebody else's compile is not ours to break.
        return True
    except (OSError, ValueError):
        return False
    return True


def _remove(entry: CacheEntry) -> str | None:
    """Delete one entry. Returns an error message, or ``None`` on success.

    A missing entry counts as success: two sweeps racing, or a ``force_recompile``
    unlinking the artifact first, both land here and both got what they wanted.
    """
    try:
        if entry.kind == KIND_SHARD:
            shutil.rmtree(entry.path)
        else:
            entry.path.unlink()
    except FileNotFoundError:
        return None
    except OSError as exc:
        if exc.errno == errno.ENOENT:
            return None
        return exc.strerror or str(exc)
    return None


def _sweep(
    selected: Sequence[CacheEntry],
    *,
    total_bytes_before: int,
    min_age: float,
    dry_run: bool,
    now: float,
) -> CacheSweep:
    """Remove ``selected``, holding back anything too young or still in use.

    The single choke point for every deletion in this module: the ``is_removable``
    assertion, the age floor, and the liveness check all live here, so no verb can
    be written that quietly skips one of them.

    The floor is measured from ``last_used``, not from ``mtime``, and the difference
    is the whole point of having one. The hazard is the window between
    ``get_cached_so`` handing back a path and the loader opening it, so what has to
    be recent is the *use*, not the build — an artifact compiled last month and
    ``dlopen``ed a minute ago is precisely the one an LRU pass over an all-fresh
    cache would nominate, and an mtime floor would wave it through. Where the
    filesystem does not record access times the two are equal and this costs nothing.
    """
    removed: list[CacheEntry] = []
    held: list[CacheEntry] = []
    failed: list[tuple[Path, str]] = []
    for entry in selected:
        if not entry.is_removable:  # pragma: no cover - callers filter first
            raise AssertionError(f"refusing to remove a foreign entry: {entry.path}")
        if now - entry.last_used < min_age or _compile_may_be_running(entry):
            held.append(entry)
            continue
        if dry_run:
            removed.append(entry)
            continue
        err = _remove(entry)
        if err is None:
            removed.append(entry)
        else:
            failed.append((entry.path, err))
    return CacheSweep(
        dry_run=dry_run,
        removed=tuple(removed),
        held=tuple(held),
        failed=tuple(failed),
        total_bytes_before=total_bytes_before,
    )


def clean_codegen_cache(
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    min_age: float | str = DEFAULT_MIN_AGE,
    dry_run: bool = False,
) -> CacheSweep:
    """Remove leaked compile partials — and only those.

    Safe by construction: the selection is exactly :data:`PARTIAL_KINDS`, so no
    compiled artifact is touched and no cache hit is lost. What it collects is the
    debris of compiles that died between writing their scratch files and the
    ``finally`` that removes them: ``bngsim_shard_*`` directories full of ``.o``, the
    ``rhs_<key>_<hash>.<pid>_<n>.c`` beside them, and (on Windows) the ``.lib``/``.exp``
    sidecars ``compile_rhs`` never unlinks.

    Parameters
    ----------
    cache_dir
        Which cache to sweep. Default: the live one (see :func:`codegen_cache_dir`).
    min_age
        Seconds, or a duration string like ``"90m"``. Nothing used or written more
        recently than this is removed. See :data:`DEFAULT_MIN_AGE`.
    dry_run
        Select and report, delete nothing.
    """
    min_age = parse_duration(min_age)
    info = codegen_cache_info(cache_dir)
    return _sweep(
        [e for e in info.entries if e.is_partial],
        total_bytes_before=info.total_bytes,
        min_age=min_age,
        dry_run=dry_run,
        now=time.time(),
    )


def prune_codegen_cache(
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    orphaned: bool = False,
    keep_keys: Sequence[str | None] = (),
    older_than: float | str | None = None,
    max_size: int | str | None = None,
    min_age: float | str = DEFAULT_MIN_AGE,
    dry_run: bool = False,
) -> CacheSweep:
    """Bound the cache by orphan status, age, size, or any combination of them.

    At least one of ``orphaned`` / ``older_than`` / ``max_size`` is required — a
    prune with no bound is either a no-op or a :func:`clear_codegen_cache`, and
    guessing which is not this function's business.

    ``orphaned=True`` is the targeted one, and usually the one wanted after an
    emitter edit: it evicts every artifact whose name carries a codegen key other
    than this install's (:attr:`CacheInfo.orphaned`), which is exactly the set no
    run can hit again. An age bound cannot express that — it keeps orphans that
    happen to be recent and throws away live artifacts that merely have not been
    loaded lately.

    Order of operations, and the one surprise worth stating: **prune subsumes clean.**
    The partial sweep runs first, then artifacts are evicted. That is what makes
    ``max_size`` mean what a user means by it — the cap is on the whole directory, so
    debris counting toward it has to be collectable before artifacts are thrown away
    to make room for it.

    ``max_size`` may be unreachable: foreign entries are never removed, and neither
    are artifacts inside the ``min_age`` floor. The sweep then evicts everything it
    legitimately can and stops. Compare ``sweep.total_bytes_after`` against the cap to
    detect it (the CLI does, and says so).

    Parameters
    ----------
    orphaned
        Evict every artifact whose codegen key is not this install's, including the
        pre-#363 names that carry no key at all. Bounded by ``keep_keys``.
    keep_keys
        Additional codegen keys to treat as live, for a cache directory shared by
        several bngsim installs — a venv per project is ordinary, and each has its
        own key, so what is orphaned from one venv's point of view is live from
        another's. ``None`` in this sequence spares the pre-#363 names (the artifacts
        of an install too old to write a key), which is the same situation one
        version further back. Only meaningful with ``orphaned=True``; passing it
        without raises, rather than reading as a protection the age and size bounds
        do not in fact honor.

        An explicit list rather than a registry of live keys maintained beside the
        cache: such a registry would have to be written by every install that ever
        compiles — including ones that never run this CLI — and a stale entry either
        protects a dead corpus forever or, in the direction that costs something,
        goes missing and authorizes deleting a live one. The flag is a decision the
        person sweeping makes with :func:`codegen_cache_info`'s per-key table in
        front of them.
    older_than
        Seconds, or a duration string like ``"30d"``. Artifacts not used within this
        window are evicted. "Used" is ``max(atime, mtime)`` — see :func:`_stamp`.
    max_size
        Bytes, or a size string like ``"2G"`` (powers of 1024). After the orphan and
        age passes, artifacts are evicted least-recently-used until the *directory*
        fits.
    min_age
        The floor below which nothing is removed, whatever the other bounds say. It
        applies to the LRU pass too, which is where it earns its keep: on a cache
        whose entries have all been loaded recently and which is over the cap, LRU
        order still has to nominate one of them — and the one it nominates may be
        the artifact a sibling process is about to ``dlopen``. It applies to the
        orphan pass as well: an orphan minutes old is one another process just
        compiled under its own key and is about to ``dlopen``.
    dry_run
        Select and report, delete nothing.
    """
    if older_than is None and max_size is None and not orphaned:
        raise ValueError(
            "prune needs a bound: pass orphaned=True, older_than= (e.g. '30d'), "
            "max_size= (e.g. '2G'), or a combination"
        )
    if keep_keys and not orphaned:
        raise ValueError("keep_keys= applies to the orphan pass; pass orphaned=True with it")
    min_age_s = parse_duration(min_age)
    older_than_s = None if older_than is None else parse_duration(older_than)
    max_size_b = None if max_size is None else parse_size(max_size)
    now = time.time()

    info = codegen_cache_info(cache_dir)
    selected: list[CacheEntry] = [e for e in info.entries if e.is_partial]
    chosen = set(selected)

    # Least-recently-used first, tie-broken by path so two runs over the same cache
    # evict the same entries in the same order.
    artifacts = sorted(
        (e for e in info.entries if e.is_artifact),
        key=lambda e: (e.last_used, str(e.path)),
    )

    if orphaned:
        # Before the age and size passes, so what they see is the cache as it will
        # be: on a developer machine the orphan sweep alone usually brings the
        # directory under a --max-size cap, and the LRU pass then has no live
        # artifact left to nominate.
        keep = {info._key_field, *keep_keys}
        for e in artifacts:
            if e.codegen_key not in keep and e not in chosen:
                selected.append(e)
                chosen.add(e)

    if older_than_s is not None:
        cutoff = now - older_than_s
        for e in artifacts:
            if e.last_used < cutoff and e not in chosen:
                selected.append(e)
                chosen.add(e)

    if max_size_b is not None:
        # Everything already selected is assumed gone; anything the floor will hold
        # back is not, but that is unknowable here and _sweep reconciles it. The
        # cap is therefore a target rather than a guarantee, which is documented.
        projected = info.total_bytes - sum(e.size for e in chosen)
        for e in artifacts:
            if projected <= max_size_b:
                break
            if e in chosen:
                continue
            selected.append(e)
            chosen.add(e)
            projected -= e.size

    # A decline note (issue #438) is worth nothing without the artifact it
    # describes and describes nothing once that artifact is gone, so it goes
    # exactly when its artifact goes. Last, so it sees every pass's selection; it
    # frees no meaningful space itself, which is why it is not in the size pass.
    kept = {_note_family(e.path.name) for e in info.entries if e.is_artifact and e not in chosen}
    for e in info.entries:
        if e.kind == KIND_NOTE and _note_family(e.path.name) not in kept:
            selected.append(e)
            chosen.add(e)

    return _sweep(
        selected,
        total_bytes_before=info.total_bytes,
        min_age=min_age_s,
        dry_run=dry_run,
        now=now,
    )


def clear_codegen_cache(
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    min_age: float | str = 0.0,
    dry_run: bool = False,
) -> CacheSweep:
    """Remove every artifact and partial bngsim owns. Foreign entries are left alone.

    ``min_age`` defaults to ``0`` here, unlike the other verbs: "clear" is an explicit
    instruction to empty the cache, and a floor that quietly left the newest entries
    behind would make it a lie. Pass one to reinstate it —
    ``clear_codegen_cache(min_age=DEFAULT_MIN_AGE)`` is the belt-and-braces form.

    The one thing a zero floor does *not* switch off is the POSIX liveness check: a
    partial whose compile is still running stays, because emptying a cache is a
    reason to reclaim disk and not a reason to break a build that is in progress. It
    is reported in ``sweep.held`` rather than silently skipped.

    The cache directory itself is not removed: ``compile_rhs`` recreates it on demand,
    but a caller that pointed ``BNGSIM_CODEGEN_CACHE_DIR`` at a directory somebody
    else provisioned (mode, ACLs, a mount point) should get it back intact.

    Examples
    --------
    >>> import bngsim
    >>> sweep = bngsim.clear_codegen_cache(dry_run=True)   # what would go
    >>> sweep.dry_run
    True
    """
    info = codegen_cache_info(cache_dir)
    return _sweep(
        [e for e in info.entries if e.is_removable],
        total_bytes_before=info.total_bytes,
        min_age=parse_duration(min_age),
        dry_run=dry_run,
        now=time.time(),
    )


# ─── Argument value parsing ──────────────────────────────────────────────────

_DURATION_UNITS: dict[str, float] = {
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
    "w": 604800.0,
}

_DURATION_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([smhdw])?\s*$", re.I)


def parse_duration(value: float | str) -> float:
    """Parse ``"30d"`` / ``"12h"`` / ``"90m"`` / ``"45s"`` / ``"2w"`` to seconds.

    A bare number is **days**, not seconds. That is the reading of ``--older-than 30``
    a user intends, and the CLI echoes the resolved cutoff as a date so a misreading
    cannot pass silently. Numeric (non-string) input is already seconds — it comes
    from the API, where ``older_than=DEFAULT_MIN_AGE`` has to mean what it says.
    """
    if not isinstance(value, str):
        seconds = float(value)
        if seconds < 0:
            raise ValueError("duration must not be negative")
        return seconds
    m = _DURATION_RE.match(value)
    if m is None:
        raise ValueError(
            f"cannot parse duration {value!r}: expected a number with an optional "
            "s/m/h/d/w suffix (e.g. '30d', '12h', '90m'); a bare number means days"
        )
    return float(m.group(1)) * _DURATION_UNITS[(m.group(2) or "d").lower()]


_SIZE_UNITS: dict[str, int] = {
    "": 1,
    "b": 1,
    "k": 1024,
    "m": 1024**2,
    "g": 1024**3,
    "t": 1024**4,
}

_SIZE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([kmgt])?(i?b)?\s*$", re.I)


def parse_size(value: int | str) -> int:
    """Parse ``"2G"`` / ``"500MB"`` / ``"1.5GiB"`` / ``"1048576"`` to bytes.

    K/M/G/T are powers of 1024 throughout — ``KiB``/``MiB``/``GiB``/``TiB`` — because
    that is what ``du -sh`` prints and this feature exists because somebody read a
    ``du``. Suffix-less input is bytes; there is no plausible alternative reading.
    """
    if not isinstance(value, str):
        # Compare before truncating: int(-0.5) is 0, so a post-conversion check would
        # wave a negative cap through as "keep nothing" and evict the whole cache.
        if value < 0:
            raise ValueError("size must not be negative")
        return int(value)
    m = _SIZE_RE.match(value)
    if m is None:
        raise ValueError(
            f"cannot parse size {value!r}: expected a number with an optional "
            "K/M/G/T suffix (e.g. '2G', '500MB', '1.5GiB'); a bare number means bytes"
        )
    return int(float(m.group(1)) * _SIZE_UNITS[(m.group(2) or "").lower()])


# ─── Report formatting ───────────────────────────────────────────────────────


def _fmt_size(n: int) -> str:
    """Bytes as a short binary-prefix string, matching ``du -h``'s scale."""
    if n < 1024:
        return f"{n} B"
    size = float(n)
    for unit in ("KiB", "MiB", "GiB", "TiB"):
        size /= 1024.0
        if size < 1024.0 or unit == "TiB":
            return f"{size:,.1f} {unit}"
    raise AssertionError  # pragma: no cover - unreachable, TiB is terminal


def _fmt_date(t: float | None) -> str:
    return "-" if t is None else datetime.fromtimestamp(t).strftime("%Y-%m-%d")


def _fmt_age(t: float | None, now: float) -> str:
    if t is None:
        return ""
    days = (now - t) / 86400.0
    if days < 1.0:
        return " (today)"
    return f" ({days:.0f} day{'s' if round(days) != 1 else ''} ago)"


#: How many keys the text report lists before summarizing the tail. A machine that
#: tracks bngsim development accumulates one key per emitter edit, so the full list
#: can run to dozens — and the reader's question ("what is holding the space, and is
#: any of it mine?") is answered by the largest few plus this install's.
_KEY_TABLE_ROWS = 8

#: What the pre-#363 bucket is called in the table, and the token ``--keep-key``
#: takes to spare it. A single ``-`` because that is what the table shows, and
#: because argparse reads a lone dash as a value rather than as an option.
_NO_KEY_LABEL = "-"


def _key_table(info: CacheInfo) -> list[str]:
    """The per-key artifact breakdown, largest first, with this install's key marked.

    Empty when there is nothing to attribute. With one key it is a single row that
    says so plainly; with several it is the audit of a shared or pre-warmed artifact
    directory — which bngsim versions' artifacts are in it, and what each holds.
    """
    by_key = info.by_key
    if not by_key:
        return []
    field = info._key_field
    rows = sorted(by_key.items(), key=lambda kv: (-kv[1][1], kv[0] or ""))
    head, tail = rows[:_KEY_TABLE_ROWS], rows[_KEY_TABLE_ROWS:]
    width = max(len(k or _NO_KEY_LABEL) for k, _ in head) + 7  # room for " (live)"
    lines = ["", f"  {'codegen key':<{width}} {'entries':>9} {'size':>12}"]
    lines.append(f"  {'-' * width} {'-' * 9} {'-' * 12}")
    for key, (n, b) in head:
        label = (key or _NO_KEY_LABEL) + (" (live)" if key == field else "")
        lines.append(f"  {label:<{width}} {n:>9,} {_fmt_size(b):>12}")
    if tail:
        n = sum(c for _, (c, _) in tail)
        b = sum(s for _, (_, s) in tail)
        lines.append(f"  {f'… {len(tail)} more key(s)':<{width}} {n:>9,} {_fmt_size(b):>12}")
    return lines


def _format_info(info: CacheInfo, *, now: float) -> str:
    lines = [f"codegen cache: {info.path}"]
    if not info.exists:
        lines.append("  (does not exist — nothing has been compiled with this cache yet)")
        return "\n".join(lines)

    built, used = info.built_span, info.used_span
    lines.append(f"  entries:   {len(info.entries):,}")
    lines.append(f"  size:      {_fmt_size(info.total_bytes)}")
    lines.append(
        f"  built:     {_fmt_date(built[0])}{_fmt_age(built[0], now)}"
        f" .. {_fmt_date(built[1])}{_fmt_age(built[1], now)}"
    )
    if info.atime_is_live:
        lines.append(
            f"  last used: {_fmt_date(used[0])}{_fmt_age(used[0], now)}"
            f" .. {_fmt_date(used[1])}{_fmt_age(used[1], now)}"
        )
    elif info.entries:
        lines.append("  last used: not recorded by this filesystem — prune orders by build time")
    lines.append(f"  key:       {info.codegen_key}")
    lines.append(f"  live:      {len(info.live):,} artifact(s), {_fmt_size(info.live_bytes)}")
    lines.append(
        f"  orphaned:  {len(info.orphaned):,} artifact(s), {_fmt_size(info.orphaned_bytes)}"
    )

    by_kind = info.by_kind
    lines.append("")
    lines.append(f"  {'kind':<15} {'entries':>9} {'size':>12}")
    lines.append(f"  {'-' * 15} {'-' * 9} {'-' * 12}")
    for kind in KINDS:
        n, b = by_kind[kind]
        lines.append(f"  {_KIND_LABELS[kind]:<15} {n:>9,} {_fmt_size(b):>12}")

    lines.extend(_key_table(info))

    notes: list[str] = []
    n_partial = sum(by_kind[k][0] for k in PARTIAL_KINDS)
    if n_partial:
        notes.append(
            f"{n_partial:,} leaked partial(s) hold {_fmt_size(info.partial_bytes)}; "
            "`bngsim-cache clean` removes them."
        )
    n_foreign = by_kind[KIND_FOREIGN][0]
    if n_foreign:
        notes.append(
            f"{n_foreign:,} entr{'y' if n_foreign == 1 else 'ies'} in this directory "
            f"{'was' if n_foreign == 1 else 'were'} not written by bngsim; no verb "
            "here will remove them."
        )
    if info.orphaned:
        notes.append(
            f"{len(info.orphaned):,} artifact(s) hold {_fmt_size(info.orphaned_bytes)} "
            "under a codegen key that is not this install's; `bngsim-cache prune "
            "--orphaned` removes exactly those. If another bngsim install shares this "
            "directory, spare its key with `--keep-key`."
        )
    if notes:
        lines.append("")
        for note in notes:
            lines.extend(
                textwrap.wrap(note, width=76, initial_indent="  ", subsequent_indent="  ")
            )
    return "\n".join(lines)


#: Keys named individually in a sweep report before it summarizes the rest.
_SWEEP_KEYS_SHOWN = 4


def _format_sweep(
    sweep: CacheSweep, *, verb: str, max_size: int | None = None, orphaned: bool = False
) -> str:
    what = "would remove" if sweep.dry_run else "removed"
    lines = [
        f"{verb}: {what} {len(sweep.removed):,} entr"
        f"{'y' if len(sweep.removed) == 1 else 'ies'}, {_fmt_size(sweep.removed_bytes)}"
        f"  ({_fmt_size(sweep.total_bytes_before)} -> {_fmt_size(sweep.total_bytes_after)})"
    ]
    if orphaned:
        # Whose artifacts went, by name. The one thing a user cannot check afterwards
        # is what a sweep took, and on a shared cache "which install did I just make
        # recompile?" is the question a wrong --keep-key raises.
        keys = sorted({e.codegen_key or _NO_KEY_LABEL for e in sweep.removed if e.is_artifact})
        if keys:
            shown = ", ".join(keys[:_SWEEP_KEYS_SHOWN])
            more = (
                f" (+{len(keys) - _SWEEP_KEYS_SHOWN} more)"
                if len(keys) > _SWEEP_KEYS_SHOWN
                else ""
            )
            lines.append(f"  orphaned artifacts under {len(keys):,} key(s): {shown}{more}")
    if sweep.held:
        lines.append(
            f"  held back {len(sweep.held):,}: too new, or a compile still holds them (--min-age)"
        )
    if max_size is not None and sweep.total_bytes_after > max_size:
        lines.append(
            f"  still over the {_fmt_size(max_size)} cap — what remains is inside "
            "--min-age or is not bngsim's to remove"
        )
    for path, err in sweep.failed:
        lines.append(f"  error: {path.name}: {err}")
    return "\n".join(lines)


# ─── Command line ────────────────────────────────────────────────────────────


def _duration_arg(value: str) -> float:
    try:
        return parse_duration(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _size_arg(value: str) -> int:
    try:
        return parse_size(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


_CACHE_DIR_HELP = (
    "operate on this directory instead of the configured cache (e.g. to inspect a "
    "shared pre-warmed artifact directory from a login node). Accepted on either "
    "side of the verb"
)


def _add_cache_dir(p: argparse.ArgumentParser, *, before_verb: bool) -> None:
    """Add ``-C/--cache-dir`` to the top-level parser and to every subparser.

    Both positions, because ``bngsim-cache info -C /scratch`` is what a user types
    and ``git``-style options-before-subcommand is what argparse gives you for free.
    The subparser copies default to ``SUPPRESS`` rather than ``None``: a real default
    there would be applied *after* the top-level parse and would silently overwrite
    a ``-C`` given before the verb with ``None``.
    """
    p.add_argument(
        "-C",
        "--cache-dir",
        type=Path,
        default=None if before_verb else argparse.SUPPRESS,
        metavar="DIR",
        help=_CACHE_DIR_HELP if before_verb else argparse.SUPPRESS,
    )


def _add_common(p: argparse.ArgumentParser, *, min_age: bool = True) -> None:
    p.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="report what would be removed; delete nothing",
    )
    if min_age:
        p.add_argument(
            "--min-age",
            type=_duration_arg,
            default=DEFAULT_MIN_AGE,
            metavar="DURATION",
            help=(
                "never remove anything used or written more recently than this, so a compile "
                "in flight — which writes its scratch files into this very directory — "
                "cannot be swept out from under itself. Default: 1h (the default "
                "BNGSIM_CODEGEN_TIMEOUT is 600 s, so an hour clears any default build)"
            ),
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bngsim-cache",
        description=(
            "Inspect and prune the code-generated artifact cache "
            "(~/.cache/bngsim/codegen, or $BNGSIM_CODEGEN_CACHE_DIR). Nothing prunes "
            "automatically: bngsim never deletes from this directory as a side effect "
            "of being used, so it grows until something here is run."
        ),
        epilog=(
            "Only files bngsim wrote are ever removed — anything else in the directory "
            "is reported and left alone."
        ),
    )
    _add_cache_dir(p, before_verb=True)
    sub = p.add_subparsers(dest="cmd", required=True, metavar="{info,clean,prune,clear}")

    info = sub.add_parser(
        "info",
        help="report path, size, dates, live vs orphaned, and breakdowns by kind and key",
        description=(
            "Census the cache. Reads directory metadata only, so it stays fast on a "
            "cache with tens of thousands of entries. Live and orphaned are read off "
            "the codegen key each artifact's name carries; `prune --orphaned` is the "
            "sweep for the second number."
        ),
    )
    _add_cache_dir(info, before_verb=False)
    info.add_argument(
        "--json",
        action="store_true",
        help="emit the report as JSON (stable keys, for scripts and dashboards)",
    )

    clean = sub.add_parser(
        "clean",
        help="remove leaked compile partials only; no compiled artifact is touched",
        description=(
            "Remove the debris of compiles that were killed before their cleanup ran: "
            "bngsim_shard_* scratch directories, the rhs_<key>_<hash>.<pid>_<n>.c beside "
            "them, and the temp/sidecar libraries. Safe by construction — the "
            "selection excludes every usable artifact, so no cache hit is lost."
        ),
    )
    _add_cache_dir(clean, before_verb=False)
    _add_common(clean)

    prune = sub.add_parser(
        "prune",
        help="drop orphaned artifacts, and/or bound the cache by age and total size",
        description=(
            "Evict artifacts to fit an orphan, age and/or size bound, "
            "least-recently-used first. Prune subsumes clean: leaked partials are "
            "swept first, so the --max-size cap is on the whole directory rather than "
            "on artifacts alone. At least one bound is required."
        ),
    )
    _add_cache_dir(prune, before_verb=False)
    prune.add_argument(
        "--orphaned",
        action="store_true",
        help=(
            "evict every artifact built under a codegen key other than this install's "
            "— what `info` reports as orphaned, and what no run here can load again. "
            "The targeted sweep after an emitter edit, where --older-than would keep "
            "recent orphans and throw away live artifacts merely idle for a while"
        ),
    )
    prune.add_argument(
        "--keep-key",
        action="append",
        default=[],
        metavar="KEY",
        help=(
            "spare this codegen key during --orphaned, for a cache directory shared "
            "by several bngsim installs (a venv per project each has its own key, so "
            "one venv's orphans are another's live artifacts). Repeatable; the keys "
            "are the ones `info` lists, and a bare - spares the entries predating "
            "keyed names"
        ),
    )
    prune.add_argument(
        "--older-than",
        type=_duration_arg,
        default=None,
        metavar="DURATION",
        help=(
            "evict artifacts not used within this window, e.g. 30d, 12h, 2w. A bare "
            "number means days. 'Used' is the newer of access and modification time; "
            "on a filesystem that does not record access times this is build time"
        ),
    )
    prune.add_argument(
        "--max-size",
        type=_size_arg,
        default=None,
        metavar="SIZE",
        help=(
            "evict least-recently-used artifacts until the directory fits, e.g. 2G, "
            "500MB, 1.5GiB. K/M/G/T are powers of 1024; a bare number is bytes"
        ),
    )
    _add_common(prune)

    clear = sub.add_parser(
        "clear",
        help="remove every artifact and partial bngsim owns",
        description=(
            "Empty the cache. Every model will recompile on its next run, which on a "
            "large corpus is minutes to hours of cc — this is the blunt instrument, "
            "and prune is usually the one you want."
        ),
    )
    _add_cache_dir(clear, before_verb=False)
    clear.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="do not prompt for confirmation (required when stdin is not a terminal)",
    )
    _add_common(clear, min_age=False)
    return p


def _confirm(info: CacheInfo, *, assume_yes: bool) -> bool:
    """Gate ``clear`` behind an explicit yes.

    Non-interactive callers must pass ``--yes`` rather than being auto-confirmed:
    a cron job or a CI step that reaches ``clear`` by accident should fail visibly,
    not silently throw away an afternoon of compiles.
    """
    if assume_yes:
        return True
    n = sum(1 for e in info.entries if e.is_removable)
    size = _fmt_size(sum(e.size for e in info.entries if e.is_removable))
    if not sys.stdin.isatty():
        print(
            f"error: clear would remove {n:,} entries ({size}) from {info.path}.\n"
            "       stdin is not a terminal, so pass --yes to confirm.",
            file=sys.stderr,
        )
        return False
    reply = input(f"Remove {n:,} entries ({size}) from {info.path}? [y/N] ")
    return reply.strip().lower() in {"y", "yes"}


def main(argv: list[str] | None = None) -> int:
    """``bngsim-cache`` / ``python -m bngsim.cache``. Returns the process exit code."""
    args = _build_parser().parse_args(argv)
    # `getattr`, not `args.cache_dir`: the post-verb copy of -C is SUPPRESS-defaulted
    # so it cannot clobber a pre-verb one, which also means it may simply be absent.
    cache_dir = getattr(args, "cache_dir", None)

    if args.cmd == "info":
        info = codegen_cache_info(cache_dir)
        if args.json:
            print(json.dumps(info.to_dict(), indent=2))
        else:
            print(_format_info(info, now=time.time()))
        return 0

    if args.cmd == "clean":
        sweep = clean_codegen_cache(cache_dir, min_age=args.min_age, dry_run=args.dry_run)
        print(_format_sweep(sweep, verb="clean"))
        return 1 if sweep.failed else 0

    if args.cmd == "prune":
        if args.older_than is None and args.max_size is None and not args.orphaned:
            print(
                "error: prune needs a bound: --orphaned, --older-than (e.g. 30d), "
                "--max-size (e.g. 2G), or a combination.",
                file=sys.stderr,
            )
            return 2
        if args.keep_key and not args.orphaned:
            print(
                "error: --keep-key applies to the orphan pass; pass --orphaned with it.",
                file=sys.stderr,
            )
            return 2
        sweep = prune_codegen_cache(
            cache_dir,
            orphaned=args.orphaned,
            # A bare `-` is how `info` renders the pre-keyed-name bucket, so it is
            # what spares it here too.
            keep_keys=[None if k == _NO_KEY_LABEL else k for k in args.keep_key],
            older_than=args.older_than,
            max_size=args.max_size,
            min_age=args.min_age,
            dry_run=args.dry_run,
        )
        print(_format_sweep(sweep, verb="prune", max_size=args.max_size, orphaned=args.orphaned))
        return 1 if sweep.failed else 0

    # clear
    info = codegen_cache_info(cache_dir)
    if not args.dry_run and not _confirm(info, assume_yes=args.yes):
        return 1
    sweep = clear_codegen_cache(cache_dir, dry_run=args.dry_run)
    print(_format_sweep(sweep, verb="clear"))
    return 1 if sweep.failed else 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
