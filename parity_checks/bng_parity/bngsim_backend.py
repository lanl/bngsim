#!/usr/bin/env python3
"""Reusable BNGsim-backend assertions for the bng_parity runner.

Guards the silent "wrong engine" failure: a parity / golden / benchmark run that
*thinks* it exercised bngsim but actually ran the legacy BNG2.pl → run_network /
NFsim subprocess stack — producing a result that never saw bngsim, with no error.

The sweep no longer drives bngsim through ``bionetgen.run(simulator='bngsim')``:
with a stock BNG2.pl that routes a BNGL ``simulate`` back to run_network/NFsim
(GH #175). It drives ``_bng_common.run_bngsim_job`` instead — BNG2.pl generates the
network/XML, then bngsim simulates IN-PROCESS via the bridge's DIRECT route. So the
engine a job uses is decided by the model's own simulate methods, not by any bridge
routing pass: bngsim runs every supported method (ode/ssa/psa, and the network-free
nf/rm — nf BNGL is run on RuleMonkey), and a job whose ONLY method is unsupported
(``pla``) is SKIPPED rather than silently run on the legacy stack and mislabelled.

Two layers:
  * ``backend_status()`` — is bngsim wired into the *active* PyBioNetGen at all
    (importable AND version >= the bridge's floor)? The global gate, used by the
    sweep engine pre-flight and recorded in run/golden provenance.
  * ``predict_engine(path)`` / ``audit_unit_engines(units)`` — which engine each
    job will use, classified the SAME way ``run_bngsim_job`` decides it
    (``_bng_common.classify_bngsim_track``). Lets a sweep audit every unit up front
    and surface (or, under --strict-engine, abort on) any job bngsim cannot run.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

# Engine labels we expose for the direct-drive audit.
ENGINE_BNGSIM = "bngsim"  # in-process bngsim (ode/ssa/psa/nf/rm) — the engine we validate
ENGINE_UNSUPPORTED = "unsupported"  # only methods bngsim cannot run (pla); the job is SKIPPED
ENGINE_UNKNOWN = "unknown"  # classifier unavailable (e.g. model unreadable)


def _load_bng_common():
    """Import the sibling ``_bng_common`` (the genuine-bngsim driver + classifier)."""
    import sys

    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import _bng_common

    return _bng_common


def _bridge():
    """Import the PyBioNetGen bngsim bridge module (source of the version floor).

    ``MINIMUM_BNGSIM_VERSION`` / ``BNGSIM_UNAVAILABLE_REASON`` live on
    ``bionetgen.core.tools.bngsim_bridge`` — only ``BNGSIM_AVAILABLE`` /
    ``BNGSIM_VERSION`` are re-exported at the ``bionetgen`` top level.
    """
    from bionetgen.core.tools import bngsim_bridge

    return bngsim_bridge


def backend_status() -> dict:
    """Whether bngsim is wired into the active PyBioNetGen, plus provenance.

    Always safe to read. ``available`` is the bridge's own determination
    (bngsim importable AND ``__version__`` >= ``MINIMUM_BNGSIM_VERSION``) — exactly
    the condition that makes ``simulator='bngsim'`` raise instead of falling back
    to the legacy stack. The provenance fields pin down *which* PyBioNetGen +
    bngsim produced a run, so a golden/report records what a consumer must install
    to reproduce it.
    """
    out = {
        "available": False,
        "bngsim_version": None,
        "min_bngsim_version": None,
        "reason": None,
        "bionetgen_path": None,
        "bionetgen_version": _dist_version("bionetgen"),
        "bionetgen_commit": bionetgen_commit(),
    }
    try:
        import bionetgen

        out["bionetgen_path"] = bionetgen.__file__
        out["available"] = bool(getattr(bionetgen, "BNGSIM_AVAILABLE", False))
        out["bngsim_version"] = getattr(bionetgen, "BNGSIM_VERSION", None)
    except Exception as exc:
        out["reason"] = f"bionetgen import failed: {exc}"
        return out
    try:
        br = _bridge()
        out["min_bngsim_version"] = getattr(br, "MINIMUM_BNGSIM_VERSION", None)
        if not out["available"]:
            out["reason"] = getattr(br, "BNGSIM_UNAVAILABLE_REASON", None) or "unknown"
    except Exception as exc:
        if out["reason"] is None:
            out["reason"] = f"bridge import failed: {exc}"
    return out


def predict_engine(input_path) -> tuple[str, str]:
    """``(engine, track_or_reason)`` for how the sweep will run this job.

    The sweep drives bngsim DIRECTLY (``_bng_common.run_bngsim_job``), so the
    engine is decided by the model's own simulate methods — classified by the SAME
    ``_bng_common.classify_bngsim_track`` the driver uses, not by any bridge route.
    Returns ``(ENGINE_BNGSIM, track)`` for a job bngsim runs (track is one of
    ode/ssa/psa/nf/rm), ``(ENGINE_UNSUPPORTED, reason)`` for a job it cannot run
    (only ``pla``) — which the sweep SKIPS, never silently running it on the legacy
    stack and mislabelling it bngsim (GH #175) — or ``(ENGINE_UNKNOWN, reason)``.
    """
    try:
        c = _load_bng_common()
        text = Path(input_path).read_text(errors="replace")
        track = c.classify_bngsim_track(text, stochastic=c._is_stochastic_text(text))
        if track is None:
            return (
                ENGINE_UNSUPPORTED,
                "only method(s) bngsim cannot run (e.g. pla)",
            )
        return ENGINE_BNGSIM, track
    except Exception as exc:
        return ENGINE_UNKNOWN, f"classify failed: {exc}"


def audit_unit_engines(units) -> dict:
    """Classify each unit's engine (the way the sweep actually runs it).

    One classification per unit via :func:`predict_engine`. Buckets:
      * ``bngsim``      — runs on in-process bngsim (broken out by track in
        ``by_track``). With the direct-drive path this is every job whose method
        bngsim supports — there is NO silent legacy fallback to surface anymore.
      * ``unsupported`` — declares only methods bngsim cannot run (``pla``); the
        sweep SKIPS it rather than running it on the legacy stack. ``--strict-engine``
        makes this a hard abort (the golden uses it).
      * ``unknown``     — the model could not be read/classified (investigate).
    Returns ``{counts, by_track, unsupported, unknown}``; the lists hold
    ``(unit, reason)`` tuples.
    """
    counts = {"bngsim": 0, "unsupported": 0, "unknown": 0}
    by_track: dict[str, int] = {}
    unsupported, unknown = [], []
    for unit in units:
        eng, info = predict_engine(_unit_path(unit))
        if eng == ENGINE_BNGSIM:
            counts["bngsim"] += 1
            by_track[info] = by_track.get(info, 0) + 1
        elif eng == ENGINE_UNSUPPORTED:
            counts["unsupported"] += 1
            unsupported.append((unit, info))
        else:
            counts["unknown"] += 1
            unknown.append((unit, info))
    return {
        "counts": counts,
        "by_track": by_track,
        "unsupported": unsupported,
        "unknown": unknown,
    }


def _unit_path(unit):
    """Best-effort run-path of a unit (a dict with 'run'/'bngl', or a path)."""
    if isinstance(unit, dict):
        return unit.get("run") or unit.get("bngl")
    return unit


def bionetgen_commit():
    """Resolved git commit of the active bionetgen install, or None.

    Prefers PEP 610 ``direct_url.json`` (a ``pip install git+...@<sha>`` records
    the resolved commit there), then falls back to ``git rev-parse`` for an
    editable checkout. None for a plain sdist/release install with no VCS record.
    """
    try:
        from importlib.metadata import distribution

        raw = distribution("bionetgen").read_text("direct_url.json")
        if raw:
            commit = (json.loads(raw).get("vcs_info") or {}).get("commit_id")
            if commit:
                return commit[:12]
    except Exception:
        pass
    try:
        import bionetgen

        pkg_root = Path(bionetgen.__file__).resolve().parent.parent
        r = subprocess.run(
            ["git", "-C", str(pkg_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()[:12]
    except Exception:
        pass
    return None


def _dist_version(name):
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


if __name__ == "__main__":
    # `python bngsim_backend.py` — quick env check for an operator.
    s = backend_status()
    print("BNGsim backend status:")
    for k in (
        "available",
        "bngsim_version",
        "min_bngsim_version",
        "reason",
        "bionetgen_version",
        "bionetgen_commit",
        "bionetgen_path",
    ):
        print(f"  {k:20s}: {s[k]}")
    raise SystemExit(0 if s["available"] else 1)
