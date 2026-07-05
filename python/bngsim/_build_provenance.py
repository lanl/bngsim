"""Stale-binary guard + build provenance for the compiled extension (issue #125).

Why this module exists
----------------------
The editable install serves **live Python from the source tree** but loads the
**compiled extension from a separately-built ``.so``**, and auto-rebuild is
intentionally disabled (``editable.rebuild = false``, issue #23, because ninja
is not reliably on PATH at runtime). So a developer can read correct C++ in the
editor while *executing* a binary built from older source — and every bespoke
correctness oracle that imports ``_bngsim_core`` to adjudicate "is bngsim
correct?" faithfully reports the **stale binary's** behaviour. That is not
hypothetical: it produced a false "warm-start rounding bug" verdict on GH #118.

This guard makes that failure mode loud. It is deliberately a **passive check**:
a pure mtime/string comparison with no build invocation, so it coexists with
``editable.rebuild = false`` (issue #23 — it must never shell out to ninja or
cmake). The *verdict* (stale / fresh) is decided purely by comparing the loaded
binary's mtime against the newest C++/CMake source file. The build-commit string
baked into the extension (``__build_commit__``, set by CMake) is used only for
the human-readable identity banner, never for the verdict.

Layered usage
-------------
* ``bngsim/__init__.py`` calls :func:`warn_if_stale` at import — a bulletproof,
  warning-only signal that reaches *any* consumer that does ``import bngsim``.
* ``python/tests/conftest.py`` calls :func:`format_report` + :func:`is_stale`
  in a ``pytest_configure`` preflight — fails the session by default so a stale
  binary can't quietly produce a green suite that gets committed as a verdict.
* Investigation oracles call :func:`print_identity` at startup so a stale-binary
  run is obvious in the log instead of invisible.

Environment gates
-----------------
* ``BNGSIM_NO_BUILD_CHECK=1`` — disable the guard entirely (treat as fresh). For
  exotic setups where the heuristic misfires; you are on your own.
* ``BNGSIM_ALLOW_STALE_CORE=1`` — acknowledge a known-stale binary: :func:`enforce`
  downgrades from raising to warning. The preflight then proceeds.
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# C++/CMake source extensions whose change should invalidate the binary.
_SOURCE_SUFFIXES = frozenset(
    {".cpp", ".cc", ".cxx", ".c", ".hpp", ".hh", ".hxx", ".h", ".ipp", ".inl"}
)
# Directories under the source root that contain C++ compiled into _bngsim_core.
# include/ and src/ are bngsim's own; third_party/ holds the vendored slices
# (sundials, nfsim, exprtk, rulemonkey) statically linked into the extension —
# a hand-patch there (e.g. the GH #116 compositeFunction.cpp fix) also requires
# a rebuild, so it counts.
_SOURCE_DIRS = ("src", "include", "third_party")
# Build-graph files that change the compiled output without being .cpp/.hpp.
_SOURCE_FILES = ("CMakeLists.txt",)

# mtime slack (seconds) to absorb same-second copy/stat jitter between a fresh
# build and its installed copy. Real C++ edits land minutes apart; the #118 gap
# was ~47 min, so this never masks a genuine staleness.
_MTIME_SLACK = 2.0


class StaleBinaryError(RuntimeError):
    """Raised when the loaded ``_bngsim_core`` is older than its C++ source."""


@dataclass(frozen=True)
class Provenance:
    """A snapshot of the loaded extension vs. its source tree."""

    core_path: Path | None
    core_mtime: float | None
    build_commit: str | None
    source_root: Path | None
    newest_source: Path | None
    newest_source_mtime: float | None
    head_commit: str | None

    @property
    def is_source_checkout(self) -> bool:
        """True when a C++ source tree sits next to the loaded package.

        False for an installed wheel (which ships no ``src/``), so the guard
        is a no-op there — exactly the desired behaviour for end users.
        """
        return self.source_root is not None

    @property
    def is_stale(self) -> bool:
        if self.core_mtime is None or self.newest_source_mtime is None:
            return False
        return self.newest_source_mtime > self.core_mtime + _MTIME_SLACK


def _checks_disabled() -> bool:
    return os.environ.get("BNGSIM_NO_BUILD_CHECK", "") not in ("", "0", "false", "False")


def _allow_stale() -> bool:
    return os.environ.get("BNGSIM_ALLOW_STALE_CORE", "") not in ("", "0", "false", "False")


def _core_path() -> Path | None:
    """Filesystem path of the *loaded* ``_bngsim_core`` extension, or None."""
    try:
        from bngsim import _bngsim_core  # noqa: PLC0415 (lazy by design)
    except Exception:
        return None
    file = getattr(_bngsim_core, "__file__", None)
    if not file:
        return None
    try:
        return Path(file).resolve()
    except Exception:
        return None


def _build_commit() -> str | None:
    try:
        from bngsim import _bngsim_core  # noqa: PLC0415
    except Exception:
        return None
    commit = getattr(_bngsim_core, "__build_commit__", None)
    if not commit or commit == "unknown":
        return None
    return str(commit)


def _source_root() -> Path | None:
    """Locate the bngsim C++ source root next to this module, or None.

    This module lives at ``<root>/python/bngsim/_build_provenance.py`` in a
    source checkout, so ``parents[2]`` is the root. Under the editable install
    ``__file__`` resolves to the live source via the scikit-build finder, so the
    root is the real tree with ``src/``. An installed wheel ships only
    ``python/bngsim`` (no ``src/``), so the marker check fails and the guard
    no-ops. Never raises.
    """
    try:
        root = Path(__file__).resolve().parents[2]
    except Exception:
        return None
    if (root / "CMakeLists.txt").is_file() and (root / "src").is_dir():
        return root
    return None


def _newest_source(root: Path) -> tuple[Path | None, float | None]:
    """Newest C++/CMake source under ``root`` as ``(path, mtime)``.

    Pure ``os.scandir`` walk — no git, no subprocess. The scanned set is small
    (a few hundred files); the cost is a handful of stat calls.
    """
    newest_path: Path | None = None
    newest_mtime = -1.0

    def _consider(p: str, mtime: float) -> None:
        nonlocal newest_path, newest_mtime
        if mtime > newest_mtime:
            newest_mtime = mtime
            newest_path = Path(p)

    def _walk(directory: str) -> None:
        try:
            entries = list(os.scandir(directory))
        except OSError:
            return
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    _walk(entry.path)
                elif (
                    entry.is_file(follow_symlinks=False)
                    and Path(entry.name).suffix.lower() in _SOURCE_SUFFIXES
                ):
                    _consider(entry.path, entry.stat().st_mtime)
            except OSError:
                continue

    for rel in _SOURCE_DIRS:
        d = root / rel
        if d.is_dir():
            _walk(str(d))
    for rel in _SOURCE_FILES:
        f = root / rel
        try:
            if f.is_file():
                _consider(str(f), f.stat().st_mtime)
        except OSError:
            continue

    if newest_path is None:
        return None, None
    return newest_path, newest_mtime


def _head_commit(root: Path) -> str | None:
    """Best-effort current ``HEAD`` short SHA for the banner only.

    Never part of the staleness verdict (keeps the guard a pure mtime check).
    Wrapped + timed out so it can never hang or raise.
    """
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return out or None


def gather(*, include_head: bool = True) -> Provenance:
    """Collect a :class:`Provenance` snapshot. Never raises."""
    core_path = _core_path()
    core_mtime: float | None = None
    if core_path is not None:
        try:
            core_mtime = core_path.stat().st_mtime
        except OSError:
            core_mtime = None

    root = _source_root()
    newest_path: Path | None = None
    newest_mtime: float | None = None
    head: str | None = None
    if root is not None:
        newest_path, newest_mtime = _newest_source(root)
        if include_head:
            head = _head_commit(root)

    return Provenance(
        core_path=core_path,
        core_mtime=core_mtime,
        build_commit=_build_commit(),
        source_root=root,
        newest_source=newest_path,
        newest_source_mtime=newest_mtime,
        head_commit=head,
    )


def _fmt_time(mtime: float | None) -> str:
    if mtime is None:
        return "?"
    try:
        return (
            datetime.fromtimestamp(mtime, tz=timezone.utc)
            .astimezone()
            .isoformat(timespec="seconds")
        )
    except Exception:
        return str(mtime)


def _rel(root: Path | None, path: Path | None) -> str:
    if path is None:
        return "?"
    if root is not None:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
    return str(path)


def identity_line(prov: Provenance | None = None) -> str:
    """One-line identity for the loaded binary (for oracle/preflight logs)."""
    prov = prov or gather()
    if prov.core_path is None:
        return "[bngsim] _bngsim_core: NOT LOADED"
    commit = prov.build_commit or "unknown"
    head = f" HEAD={prov.head_commit}" if prov.head_commit else ""
    state = "STALE" if prov.is_stale else ("fresh" if prov.is_source_checkout else "installed")
    return (
        f"[bngsim] _bngsim_core: {prov.core_path} | "
        f"built={commit}{head} | mtime={_fmt_time(prov.core_mtime)} | {state}"
    )


def format_report(prov: Provenance | None = None) -> str:
    """Multi-line human report used in warnings and the preflight error."""
    prov = prov or gather()
    lines = [identity_line(prov)]
    if prov.is_stale:
        lines.append(
            f"[bngsim]   STALE: {_rel(prov.source_root, prov.newest_source)} "
            f"({_fmt_time(prov.newest_source_mtime)}) is newer than the loaded "
            f"binary ({_fmt_time(prov.core_mtime)})."
        )
        lines.append(
            "[bngsim]   The binary does NOT reflect current C++. Any correctness "
            "verdict drawn from it is a statement about OLD code (see GH #125)."
        )
        lines.append(
            "[bngsim]   Rebuild:  python scripts/rebuild_editable.py   "
            "(or set BNGSIM_ALLOW_STALE_CORE=1 to proceed anyway)."
        )
    return "\n".join(lines)


def is_stale() -> bool:
    """True iff a source checkout's C++ is newer than the loaded binary.

    Honors ``BNGSIM_NO_BUILD_CHECK`` (returns False when disabled).
    """
    if _checks_disabled():
        return False
    return gather(include_head=False).is_stale


def print_identity(stream=None) -> Provenance:
    """Print the binary identity banner (default: stderr). Returns the snapshot.

    The one-liner every investigation oracle should emit at startup so a
    stale-binary run is obvious rather than invisible (GH #125, item 2).
    """
    prov = gather()
    print(
        format_report(prov) if prov.is_stale else identity_line(prov),
        file=stream or sys.stderr,
        flush=True,
    )
    return prov


def warn_if_stale() -> None:
    """Emit a warning if the loaded binary is stale. Never raises.

    Wired into ``bngsim/__init__.py`` so the signal reaches any ``import bngsim``
    consumer for free. Warning-only and fully guarded — it can never break an
    import, no matter how exotic the environment.
    """
    try:
        if _checks_disabled():
            return
        prov = gather(include_head=False)
        if prov.is_stale:
            warnings.warn(format_report(prov), RuntimeWarning, stacklevel=2)
    except Exception:
        # A provenance check must never be the reason an import fails.
        pass


def blocking_report(prov: Provenance | None = None) -> str | None:
    """Report string if the loaded binary should BLOCK the run, else ``None``.

    Blocks when the binary is stale **and** neither escape hatch is set
    (``BNGSIM_ALLOW_STALE_CORE`` / ``BNGSIM_NO_BUILD_CHECK``). Used by the pytest
    preflight so the env-gate decision lives here, not in ``conftest``. Distinct
    from :func:`enforce` (which raises ``StaleBinaryError``) so a pytest caller
    can wrap the message in ``pytest.UsageError`` for a clean, traceback-free
    abort. Pass an already-gathered ``prov`` to avoid a second scan.
    """
    if _checks_disabled():
        return None
    prov = prov or gather(include_head=False)
    if prov.is_stale and not _allow_stale():
        return format_report(prov)
    return None


def enforce(*, context: str = "") -> Provenance:
    """Raise :class:`StaleBinaryError` if the loaded binary is stale.

    The teeth of the guard, for the pytest preflight and any oracle that wants
    to refuse to run against an out-of-date binary. ``BNGSIM_NO_BUILD_CHECK``
    skips the check; ``BNGSIM_ALLOW_STALE_CORE`` downgrades the raise to a warning.
    """
    prov = gather(include_head=False)
    if _checks_disabled() or not prov.is_stale:
        return prov
    report = format_report(prov)
    if context:
        report = f"{report}\n[bngsim]   ({context})"
    if _allow_stale():
        warnings.warn(report, RuntimeWarning, stacklevel=2)
        return prov
    raise StaleBinaryError("\n" + report)
