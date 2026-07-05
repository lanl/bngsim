"""BioModels Antimony pool for ``bench_1013_4engine`` (and related harnesses): manifest, fetch, convert.

Publication slice: first 695 ``BIOMD*.ant`` plus first 268 ``MODEL*.ant`` (lexicographic within each
prefix), consistent with the supplement publication Antimony snapshot ordering.
``review_extra_ids`` in the manifest union additional BioModels IDs outside that slice; tooling
materializes them under ``biomodels_ant/review_extra/`` so the top-level directory can stay
aligned with the publication slice file count when desired.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MANIFEST_NAME = "biomodels_ant_pool_manifest.json"
_ID_RE = re.compile(r"^(BIOMD|MODEL)\d+$")
# Review extras live in ``review_extra/``.
_REVIEW_EXTRA_SUBDIR = "review_extra"


def _biomodels_dir() -> Path:
    return Path(__file__).resolve().parent


def manifest_path() -> Path:
    return _biomodels_dir() / _MANIFEST_NAME


def _dedupe_stable_strs(raw: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in raw:
        s = str(x).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _review_extra_id_set(manifest: dict[str, Any]) -> set[str]:
    """IDs listed under ``review_extra_ids`` (preferred) or legacy ``bill_extra_ids``."""
    raw = manifest.get("review_extra_ids") or manifest.get("bill_extra_ids") or []
    return {str(x).strip() for x in raw if str(x).strip()}


def _validate_manifest(data: dict[str, Any]) -> None:
    """Raise ``ValueError`` if manifest is unusable."""
    if not isinstance(data, dict):
        raise ValueError("manifest root must be a JSON object")
    ids = data.get("all_required_ids")
    pub = data.get("publication_benchmark_ids")
    extra = data.get("review_extra_ids") or data.get("bill_extra_ids")
    if isinstance(ids, list) and ids:
        id_list = _dedupe_stable_strs(ids)
        if not id_list:
            raise ValueError("all_required_ids is empty after deduplication")
    elif isinstance(pub, list) and isinstance(extra, list):
        id_list = _dedupe_stable_strs(list(pub) + list(extra))
        if not id_list:
            raise ValueError("publication_benchmark_ids and review_extra_ids yield no IDs")
    else:
        raise ValueError(
            "manifest must contain all_required_ids (non-empty list) or "
            "both publication_benchmark_ids and review_extra_ids (lists)"
        )
    for s in id_list:
        if not _ID_RE.match(s):
            raise ValueError(f"invalid model id (expected BIOMD… / MODEL…): {s!r}")
    if isinstance(extra, list) and extra:
        ex_set = {str(b).strip() for b in extra if str(b).strip()}
        req_set = set(id_list)
        unknown = sorted(ex_set - req_set)
        if unknown and isinstance(ids, list) and ids:
            logger.warning(
                "review_extra_ids contains entries not in all_required_ids: %s",
                unknown[:8],
            )


def load_manifest() -> dict[str, Any]:
    mp = manifest_path()
    if not mp.is_file():
        raise FileNotFoundError(f"Missing {mp}. See benchmarks/biomodels_ant/README.txt.")
    with open(mp, encoding="utf-8") as f:
        data = json.load(f)
    _validate_manifest(data)
    return data


def required_ids(
    manifest: dict[str, Any] | None = None,
    *,
    include_review_extras: bool = False,
) -> list[str]:
    """IDs to treat as the benchmark pool.

    Default (**include_review_extras=False**): ``publication_benchmark_ids``.
    Current manifest aligns this list with the full ``all_required_ids`` union (~1013).
    ``include_review_extras=True`` is retained for compatibility with older manifests.
    """
    m = manifest or load_manifest()
    if include_review_extras:
        ids = m.get("all_required_ids")
        if isinstance(ids, list) and ids:
            return _dedupe_stable_strs(ids)
        pub = m.get("publication_benchmark_ids") or []
        extra = m.get("review_extra_ids") or m.get("bill_extra_ids") or []
        return _dedupe_stable_strs(list(pub) + list(extra))
    pub = m.get("publication_benchmark_ids")
    if isinstance(pub, list) and pub:
        return _dedupe_stable_strs(pub)
    raise ValueError("manifest must list publication_benchmark_ids for the default benchmark set")


def _filter_id_subset(ordered: list[str], id_subset: list[str] | None) -> list[str]:
    if not id_subset:
        return ordered
    allow = set(id_subset)
    return [mid for mid in ordered if mid in allow]


def _import_step1():
    d = _biomodels_dir()
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    import step1_fetch  # noqa: PLC0415 — runtime path setup

    return step1_fetch


def _import_config():
    d = _biomodels_dir()
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    import config  # noqa: PLC0415

    return config


def sbml_downloads_dir() -> Path:
    cfg = _import_config()
    return (_biomodels_dir() / cfg.SBML_DOWNLOADS_DIR).resolve()


def default_ant_output_dir() -> Path:
    """``bngsim/benchmarks/biomodels_ant`` (sibling of ``biomodels/``)."""
    return (_biomodels_dir().parent / "biomodels_ant").resolve()


def _ant_paths_for_id(ant_dir: Path, model_id: str, manifest: dict[str, Any]) -> tuple[Path, ...]:
    """Candidate ``.ant`` paths for ``model_id`` (first existing wins elsewhere)."""
    if model_id in _review_extra_id_set(manifest):
        return (
            ant_dir / _REVIEW_EXTRA_SUBDIR / f"{model_id}.ant",
            ant_dir / f"{model_id}.ant",
        )
    return (ant_dir / f"{model_id}.ant",)


def ant_write_path(ant_dir: Path, model_id: str, manifest: dict[str, Any]) -> Path:
    """Target path for a newly written ``.ant`` (review extras → ``review_extra/``)."""
    if model_id in _review_extra_id_set(manifest):
        return ant_dir / _REVIEW_EXTRA_SUBDIR / f"{model_id}.ant"
    return ant_dir / f"{model_id}.ant"


def ant_present(ant_dir: Path, model_id: str, manifest: dict[str, Any]) -> bool:
    """Whether an ``.ant`` exists for ``model_id`` (checks current + legacy layouts)."""
    return any(p.is_file() for p in _ant_paths_for_id(ant_dir, model_id, manifest))


def pool_readiness(
    ant_dir: Path,
    manifest: dict[str, Any] | None = None,
    *,
    include_review_extras: bool = False,
    id_subset: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return ``(missing_ant_ids, present_ant_ids)`` for the active ID set."""
    man = manifest or load_manifest()
    req = _filter_id_subset(
        required_ids(man, include_review_extras=include_review_extras),
        id_subset,
    )
    missing: list[str] = []
    present: list[str] = []
    for mid in req:
        if ant_present(ant_dir, mid, man):
            present.append(mid)
        else:
            missing.append(mid)
    return missing, present


def convert_sbml_to_ant_file(sbml_path: Path, ant_path: Path, antimony_mod) -> bool:
    """Convert one SBML file to Antimony using the ``antimony`` module API (atomic replace)."""
    antimony_mod.clearPreviousLoads()
    ret = antimony_mod.loadSBMLFile(str(sbml_path))
    if ret == -1:
        err = antimony_mod.getLastError()
        logger.warning("loadSBMLFile failed %s: %s", sbml_path.name, (err or "")[:200])
        return False
    module_names = antimony_mod.getModuleNames()
    if not module_names:
        logger.warning("No module names for %s", sbml_path.name)
        return False
    mod = module_names[-1]
    ant_str = antimony_mod.getAntimonyString(mod)
    if not ant_str:
        logger.warning("Empty antimony string for %s", sbml_path.name)
        return False
    ant_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = ant_path.with_name(ant_path.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        tmp.write_text(ant_str, encoding="utf-8")
        os.replace(str(tmp), str(ant_path))
    except OSError:
        tmp.unlink(missing_ok=True)
        raise
    return True


def fetch_sbml_for_ids(
    model_ids: list[str],
    *,
    delay_s: float | None = None,
) -> tuple[list[str], list[str]]:
    """Download SBML for ``model_ids`` into ``data/sbml_downloads``. Returns (ok, failed).

    ``step1_fetch.download_sbml`` resolves relative paths against the process working directory;
    we ``chdir`` into the ``biomodels/`` package directory for the duration so outputs land
    under that tree.
    """
    step1_fetch = _import_step1()
    cfg = _import_config()
    delay = cfg.REQUEST_DELAY if delay_s is None else delay_s
    ok: list[str] = []
    failed: list[str] = []
    root = _biomodels_dir()
    prev = os.getcwd()
    try:
        os.chdir(root)
        for i, mid in enumerate(model_ids):
            try:
                if step1_fetch.download_sbml(mid):
                    ok.append(mid)
                else:
                    failed.append(mid)
                    logger.warning("download_sbml returned False for %s", mid)
            except Exception:
                logger.exception("download_sbml raised for %s", mid)
                failed.append(mid)
            if i + 1 < len(model_ids):
                time.sleep(delay)
    finally:
        with contextlib.suppress(OSError):
            os.chdir(prev)
    return ok, failed


def resolve_manifest_ant_paths(
    ant_dir: Path,
    *,
    include_review_extras: bool = False,
    limit: int | None = None,
) -> list[Path] | None:
    """Return ordered ``Path``s for manifest IDs, or ``None`` if no manifest file.

    Raises:
        FileNotFoundError: ``ant_dir`` unusable or required ``.ant`` files absent.
        ValueError: invalid manifest (from ``load_manifest``).
    """
    if not manifest_path().is_file():
        return None
    ant_dir = ant_dir.resolve()
    if ant_dir.exists() and not ant_dir.is_dir():
        raise FileNotFoundError(f"ANT path exists but is not a directory: {ant_dir}")
    ant_dir.mkdir(parents=True, exist_ok=True)
    man = load_manifest()
    ids = required_ids(man, include_review_extras=include_review_extras)
    if limit is not None and limit > 0:
        ids = ids[:limit]
    out: list[Path] = []
    missing: list[str] = []
    for mid in ids:
        chosen: Path | None = None
        for p in _ant_paths_for_id(ant_dir, mid, man):
            if p.is_file():
                chosen = p
                break
        if chosen is not None:
            out.append(chosen)
        else:
            missing.append(mid)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} manifest model(s) missing .ant under {ant_dir}: "
            f"{missing[:12]!r}{' …' if len(missing) > 12 else ''}"
        )
    return out


def ensure_biomodels_ant_pool(
    ant_dir: Path | None = None,
    *,
    fetch_network: bool = True,
    manifest: dict[str, Any] | None = None,
    include_review_extras: bool = False,
    id_subset: list[str] | None = None,
) -> dict[str, Any]:
    """Ensure every active manifest ID has a corresponding ``.ant`` under ``ant_dir``.

    Use ``include_review_extras=True`` for the full manifest union (recommended).
    Pass ``id_subset`` to ensure only a subset (e.g. first ``N`` models for ``--quick``).

    If SBML is missing, downloads from BioModels (unless ``fetch_network=False``).
    Then converts SBML → Antimony with libAntimony.

    Returns a summary dict suitable for logging or JSON. ``status`` is ``complete`` only when
    every required model has an ``.ant`` and there were no fetch/convert failures.
    """
    ant_dir = (ant_dir or default_ant_output_dir()).resolve()
    summary: dict[str, Any] = {"ant_dir": str(ant_dir)}
    if ant_dir.exists() and not ant_dir.is_dir():
        summary["status"] = "error_invalid_ant_dir"
        summary["error"] = f"path exists but is not a directory: {ant_dir}"
        return summary
    try:
        ant_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        summary["status"] = "error_invalid_ant_dir"
        summary["error"] = str(e)
        return summary

    try:
        man = manifest or load_manifest()
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError) as e:
        summary["status"] = "error_manifest"
        summary["error"] = str(e)
        return summary

    try:
        req_ids = _filter_id_subset(
            required_ids(man, include_review_extras=include_review_extras),
            id_subset,
        )
    except ValueError as e:
        summary["status"] = "error_manifest"
        summary["error"] = str(e)
        return summary
    sbml_root = sbml_downloads_dir()
    sbml_root.mkdir(parents=True, exist_ok=True)

    missing_ant, present_ant = pool_readiness(
        ant_dir,
        man,
        include_review_extras=include_review_extras,
        id_subset=id_subset,
    )
    summary.update(
        {
            "sbml_downloads_dir": str(sbml_root),
            "required_count": len(req_ids),
            "already_present": len(present_ant),
            "missing_ant_before": len(missing_ant),
        }
    )

    if not missing_ant:
        summary["status"] = "complete"
        summary["fetch_failed"] = []
        summary["convert_failed"] = []
        summary["present_after"] = len(present_ant)
        return summary

    need_sbml: list[str] = []
    for mid in missing_ant:
        xml = sbml_root / f"{mid}.xml"
        if not xml.is_file():
            need_sbml.append(mid)

    fetch_failed: list[str] = []
    if need_sbml:
        if not fetch_network:
            summary["status"] = "incomplete_missing_sbml"
            summary["missing_sbml_ids"] = need_sbml
            summary["missing_ant_ids"] = missing_ant
            summary["fetch_failed"] = []
            summary["convert_failed"] = []
            return summary
        logger.info("Fetching %d SBML file(s) from BioModels…", len(need_sbml))
        _, fetch_failed = fetch_sbml_for_ids(need_sbml)

    try:
        import antimony as ant  # noqa: PLC0415
    except ImportError as e:
        summary["status"] = "incomplete_no_antimony"
        summary["error"] = str(e)
        summary["fetch_failed"] = fetch_failed
        summary["hint"] = "pip install antimony"
        return summary

    convert_failed: list[str] = []
    still_missing: list[str] = []
    for mid in missing_ant:
        xml = sbml_root / f"{mid}.xml"
        out_ant = ant_write_path(ant_dir, mid, man)
        out_ant.parent.mkdir(parents=True, exist_ok=True)
        if not xml.is_file():
            still_missing.append(mid)
            continue
        try:
            if convert_sbml_to_ant_file(xml, out_ant, ant):
                logger.debug("Wrote %s", out_ant)
            else:
                convert_failed.append(mid)
        except OSError:
            logger.exception("Failed writing %s", out_ant)
            convert_failed.append(mid)

    missing_ant_after, present_after = pool_readiness(
        ant_dir,
        man,
        include_review_extras=include_review_extras,
        id_subset=id_subset,
    )
    complete = (
        not missing_ant_after and not fetch_failed and not convert_failed and not still_missing
    )
    summary.update(
        {
            "status": "complete" if complete else "incomplete",
            "missing_ant_after": len(missing_ant_after),
            "present_after": len(present_after),
            "fetch_failed": fetch_failed,
            "convert_failed": convert_failed,
            "still_missing_sbml": still_missing,
        }
    )
    return summary


def pool_ensure_succeeded(summary: dict[str, Any]) -> bool:
    """True if ``ensure_biomodels_ant_pool`` fully materialized the pool."""
    return summary.get("status") == "complete"
