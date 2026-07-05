"""Engine/library version stamping. Every report records what it ran against.

Each probe is best-effort: a missing package returns ``None`` rather than
raising, so a report can be produced on a machine that lacks a given reference
engine (it just records the absence). Pinned versions live in the suite's
manifest ``_meta``; these are the *observed* versions at run time, stamped into
the report ``_meta`` and each JobResult.versions so a result is always
traceable to the exact stack that produced it.
"""

from __future__ import annotations

import platform
import subprocess
from functools import lru_cache


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except Exception:
        return None


@lru_cache(maxsize=1)
def bngsim_version() -> str | None:
    try:
        import bngsim

        return getattr(bngsim, "__version__", None)
    except Exception:
        return None


@lru_cache(maxsize=1)
def roadrunner_version() -> str | None:
    try:
        import roadrunner

        return getattr(roadrunner, "__version__", None) or _pkg_version("libroadrunner")
    except Exception:
        return None


@lru_cache(maxsize=1)
def amici_version() -> str | None:
    return _pkg_version("amici")


@lru_cache(maxsize=1)
def libsbml_version() -> str | None:
    try:
        import libsbml

        return libsbml.getLibSBMLDottedVersion()
    except Exception:
        return _pkg_version("python-libsbml")


@lru_cache(maxsize=1)
def bng_version(bng2_pl: str | None = None) -> str | None:
    """Legacy BioNetGen version, read from its VERSION file if locatable."""
    import os
    from pathlib import Path

    cand = bng2_pl or os.environ.get("BNG2_PL") or os.environ.get("BNGPATH")
    if not cand:
        return None
    p = Path(cand)
    root = p.parent if p.is_file() else p
    vfile = root / "VERSION"
    if vfile.exists():
        return vfile.read_text().strip()
    return None


# Map a reference_engine name (as used in the manifest) to its probe.
REFERENCE_PROBES = {
    "roadrunner": roadrunner_version,
    "amici": amici_version,
    "bng": bng_version,
}


def stamp(reference_engine: str | None = None) -> dict:
    """Version dict for a report: bngsim + platform + the named reference engine."""
    out = {
        "bngsim": bngsim_version(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if reference_engine:
        probe = REFERENCE_PROBES.get(reference_engine)
        out[reference_engine] = probe() if probe else None
    return out


def git_rev(repo_dir: str | None = None) -> str | None:
    """Short git revision of the repo, for report provenance."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=repo_dir,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None
