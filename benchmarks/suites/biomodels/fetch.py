#!/usr/bin/env python3
"""
BioModels SBML fetch tool -- Stage 0 of the ``biomodels`` suite.

Downloads ODE-model SBML files from the BioModels REST API into the
git-ignored ``data/sbml_downloads/`` directory, where ``filter.py``
(Stage 1) then crawls them in place.

This is the reproducibility *reconstruction* step: the corpus itself is
never committed (~273 MB); a reviewer re-runs this script to rebuild it,
then re-runs ``filter.py`` to confirm the committed ``manifest.csv``.

What "already fetched" means here is simply "an ``<id>.xml`` file is
already present on disk" -- the filesystem is the single source of
truth, so re-running this script is idempotent and resumable. The only
side-state is ``data/model_registry.json``, a 7-day cache of the ODE
model-id list (the BioModels search API omits ``modellingApproach``, so
classifying a model as ODE needs a per-model metadata crawl that is far
too slow to repeat on every invocation).

Vendored from the ``ssys`` project's ``biomodels_batch/step1_fetch.py``
(plus ``config.py`` / ``utils.py``), collapsed into one self-contained
file and repointed to write into this suite's ``data/`` directory.

Usage:
    # refresh: download every ODE model not yet on disk (default)
    python fetch.py

    # download at most N new ODE models
    python fetch.py --n 50

    # include non-ODE models too (skips the slow ODE metadata crawl)
    python fetch.py --all-models

    # force a refresh of the cached ODE model-id list
    python fetch.py --refresh-registry
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
SBML_DOWNLOADS_DIR = DATA_DIR / "sbml_downloads"
MODEL_REGISTRY = DATA_DIR / "model_registry.json"
PROVENANCE_MANIFEST = SCRIPT_DIR / "biomodels_provenance.json"

REGISTRY_TTL_SECONDS = 7 * 24 * 3600  # reuse the cached id-list for a week
METADATA_DELAY = 0.05  # polite pause between per-model metadata calls

logger = logging.getLogger("biomodels.fetch")


# --------------------------------------------------------------------------
# optional tqdm -- fall back to a transparent pass-through iterator
# --------------------------------------------------------------------------
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - cosmetic only

    def tqdm(iterable, **_kwargs):
        return iterable


# --------------------------------------------------------------------------
# corpus already on disk
# --------------------------------------------------------------------------
def _is_valid_sbml_file(p: Path) -> bool:
    """A non-empty file that actually looks like SBML (cheap header sniff, not a
    full parse). Rejects 0-byte stubs and non-SBML exports (e.g. a COPASI file)."""
    try:
        if p.stat().st_size == 0:
            return False
        return "<sbml" in p.read_bytes()[:8192].decode("utf-8", "replace").lower()
    except OSError:
        return False


def already_fetched_ids() -> set[str]:
    """Model ids whose SBML file is already present AND valid in
    ``data/sbml_downloads``. A 0-byte or non-SBML file does NOT count as fetched, so
    a previously-failed download is re-attempted next run — otherwise the corpus
    would carry empty/COPASI stubs forever (silently skipped as 'present')."""
    if not SBML_DOWNLOADS_DIR.is_dir():
        return set()
    return {p.stem for p in SBML_DOWNLOADS_DIR.glob("*.xml") if _is_valid_sbml_file(p)}


# --------------------------------------------------------------------------
# BioModels model-id discovery
# --------------------------------------------------------------------------
def _search_all_model_ids(bm) -> list[str]:
    """Page through the BioModels search API for every model id."""
    ids: list[str] = []
    offset, batch_size = 0, 100
    while True:
        result = bm.search("*", offset=offset, numResults=batch_size)
        if not result or "models" not in result:
            break
        batch = result["models"]
        if not batch:
            break
        ids.extend(m["id"] for m in batch)
        logger.info("  discovered %d model ids so far...", len(ids))
        if len(batch) < batch_size:
            break
        offset += batch_size
    return ids


def _filter_ode_model_ids(bm, model_ids: list[str]) -> list[str]:
    """Keep only ids whose ``modellingApproach`` is an ODE.

    The search API does not return ``modellingApproach``, so each model
    is probed individually -- the expensive crawl this script caches.
    """
    ode: list[str] = []
    logger.info("classifying %d models by modelling approach...", len(model_ids))
    for i, model_id in enumerate(model_ids):
        if (i + 1) % 100 == 0:
            logger.info("  checked %d/%d, %d ODE so far...", i + 1, len(model_ids), len(ode))
        try:
            info = bm.get_model(model_id)
            if not info or not isinstance(info, dict):
                continue
            approach = info.get("modellingApproach", {})
            name = (
                approach.get("name", "") if isinstance(approach, dict) else str(approach)
            ).lower()
            if "ordinary differential equation" in name:
                ode.append(model_id)
        except Exception as exc:  # many MODEL* ids 404 -- expected
            logger.debug("could not classify %s: %s", model_id, exc)
        time.sleep(METADATA_DELAY)
    logger.info("found %d ODE models out of %d", len(ode), len(model_ids))
    return ode


def available_model_ids(*, ode_only: bool, refresh_registry: bool) -> list[str]:
    """Return the BioModels model-id list, ODE-filtered, with on-disk cache."""
    cache_key = "ode_models" if ode_only else "models"

    if not refresh_registry and MODEL_REGISTRY.exists():
        registry = json.loads(MODEL_REGISTRY.read_text())
        age = time.time() - registry.get("timestamp", 0)
        if age < REGISTRY_TTL_SECONDS and cache_key in registry:
            logger.info(
                "using cached model registry (%d %s, %.1f days old)",
                len(registry[cache_key]),
                cache_key,
                age / 86400,
            )
            return registry[cache_key]

    from bioservices import BioModels  # imported lazily -- heavy + optional

    bm = BioModels()
    logger.info("querying BioModels search API for the full model list...")
    all_ids = _search_all_model_ids(bm)
    logger.info("BioModels reports %d total models", len(all_ids))

    ode_ids = _filter_ode_model_ids(bm, all_ids) if ode_only else []

    payload = {
        "timestamp": time.time(),
        "date": datetime.now().isoformat(timespec="seconds"),
        "models": all_ids,
    }
    if ode_only:
        payload["ode_models"] = ode_ids
    MODEL_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    MODEL_REGISTRY.write_text(json.dumps(payload, indent=2))

    return ode_ids if ode_only else all_ids


# --------------------------------------------------------------------------
# per-model SBML download
# --------------------------------------------------------------------------
def _looks_like_sbml(data: bytes) -> bool:
    return bool(data) and b"<sbml" in data[:8192].lower()


def _download_via_rest(model_id: str) -> bytes | None:
    """Fallback path: fetch the model's ``main`` file straight from the BioModels
    REST download endpoint. The OMEX extraction below can pick an empty or wrong
    entry for some (often non-curated) models — this names the main file explicitly
    via the metadata API, which is what successfully recovered the empties. Returns
    validated SBML bytes or ``None``."""
    import urllib.parse
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"https://www.ebi.ac.uk/biomodels/{model_id}?format=json", timeout=30
        ) as r:
            main = (json.load(r).get("files", {}) or {}).get("main", [])
        if not main:
            return None
        url = (
            "https://www.ebi.ac.uk/biomodels/model/download/"
            + model_id
            + "?"
            + (urllib.parse.urlencode({"filename": main[0]["name"]}))
        )
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        return data if _looks_like_sbml(data) else None
    except Exception as exc:
        logger.error("REST fallback failed for %s: %s", model_id, exc)
        return None


def download_sbml(bm, model_id: str) -> bool:
    """Download + extract the SBML for one model. Idempotent: skips only when a
    VALID SBML is already on disk (a 0-byte/non-SBML stub is re-fetched). Validates
    every candidate and falls back to the REST endpoint, so it never writes an empty
    or non-SBML file."""
    out_path = SBML_DOWNLOADS_DIR / f"{model_id}.xml"
    if _is_valid_sbml_file(out_path):
        return True
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # bioservices.get_model_download writes "<id>.zip" into the CWD.
    zip_path = Path(f"{model_id}.zip")
    data: bytes | None = None
    try:
        bm.get_model_download(model_id)
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                names = zf.namelist()
                # BioModels OMEX archives name the SBML "<id>_url.xml"; fall
                # back progressively to any plausible non-manifest .xml. Try EACH
                # candidate and keep the first that actually validates as SBML.
                cands = [n for n in names if n.endswith("_url.xml")]
                cands += [n for n in names if n.endswith(".xml") and "sbml" in n.lower()]
                cands += [n for n in names if n.endswith(".xml") and "manifest" not in n.lower()]
                for n in dict.fromkeys(cands):  # de-dup, keep order
                    candidate = zf.read(n)
                    if _looks_like_sbml(candidate):
                        data = candidate
                        break
    except zipfile.BadZipFile:
        logger.error("corrupt archive for %s", model_id)
    except Exception as exc:
        logger.error("get_model_download failed for %s: %s", model_id, exc)
    finally:
        if zip_path.exists():
            zip_path.unlink()

    # OMEX path yielded nothing valid → try the REST download endpoint.
    if data is None:
        data = _download_via_rest(model_id)
    if data is None:
        logger.warning("no valid SBML obtained for %s", model_id)
        return False
    out_path.write_bytes(data)
    return True


# --------------------------------------------------------------------------
# pinned (reproducible) fetch -- from the provenance manifest
# --------------------------------------------------------------------------
def _download_versioned(model_id: str, version: str, fname: str) -> bytes | None:
    """Download an exact BioModels version: ``model/download/<id>.<version>``.

    This endpoint returns the bytes of that specific revision (verified: the
    latest version reproduces our tested sha256), so the fetch is deterministic
    even after BioModels adds newer revisions."""
    url = (
        f"https://www.ebi.ac.uk/biomodels/model/download/{model_id}.{version}?"
        + urllib.parse.urlencode({"filename": fname})
    )
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        return data if _looks_like_sbml(data) else None
    except Exception as exc:  # noqa: BLE001
        logger.error("versioned download failed for %s.%s: %s", model_id, version, exc)
        return None


def fetch_pinned(manifest_path: Path, dest_dir: Path, *, verify: bool = True) -> int:
    """Reconstruct the corpus at its pinned versions and verify sha256.

    Reads the provenance manifest (``generate_provenance_manifest.py`` output),
    downloads each model's pinned version, and asserts the bytes match the
    recorded sha256. Idempotent: a valid, hash-matching file on disk is kept.
    Returns the number of drift/missing failures."""
    manifest = json.loads(manifest_path.read_text())
    records = manifest["records"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    ok = drift = unresolved = 0
    for rec in tqdm(records, desc="pinned fetch"):
        mid = rec["id"]
        want = rec["sha256"]
        version = rec["origin"]["version"]
        fname = rec.get("main_file", f"{mid}_url.xml")
        out = dest_dir / f"{mid}.xml"
        if out.is_file() and hashlib.sha256(out.read_bytes()).hexdigest() == want:
            ok += 1
            continue
        if version == "unresolved":
            logger.warning(
                "no pinned version for %s (unresolved drift) -- use the frozen archive", mid
            )
            unresolved += 1
            continue
        data = _download_versioned(mid, version, fname)
        if data is None:
            drift += 1
            continue
        got = hashlib.sha256(data).hexdigest()
        if verify and got != want:
            logger.error("DRIFT %s.%s: manifest %s got %s", mid, version, want[:12], got[:12])
            drift += 1
            continue
        out.write_bytes(data)
        ok += 1
    logger.info(
        "pinned fetch: %d ok, %d drift/failed, %d unresolved (of %d)",
        ok,
        drift,
        unresolved,
        len(records),
    )
    return drift + unresolved


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
def select_models(
    available: list[str], on_disk: set[str], *, n: int | None, strategy: str
) -> list[str]:
    """Pick which missing models to fetch this run."""
    missing = [m for m in available if m not in on_disk]
    logger.info(
        "%d available, %d already on disk, %d missing", len(available), len(on_disk), len(missing)
    )
    if not missing:
        return []
    if n is None or n >= len(missing):
        return missing
    return random.sample(missing, n) if strategy == "random" else missing[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch ODE-model SBML from BioModels")
    ap.add_argument(
        "--upstream-latest",
        action="store_true",
        help="ignore the pin and crawl BioModels for the LATEST of every ODE model "
        "(refresh/extend; unpinned). Default is the reproducible pinned fetch.",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=PROVENANCE_MANIFEST,
        help="provenance manifest for the pinned fetch (default: biomodels_provenance.json)",
    )
    ap.add_argument(
        "--n", type=int, default=None, help="cap on new models to fetch (default: all missing)"
    )
    ap.add_argument(
        "--strategy",
        choices=["random", "sequential"],
        default="sequential",
        help="selection order when --n is set (default: sequential)",
    )
    ap.add_argument(
        "--all-models",
        action="store_true",
        help="include non-ODE models (skips the slow ODE metadata crawl)",
    )
    ap.add_argument(
        "--refresh-registry",
        action="store_true",
        help="ignore the cached model-id list and re-crawl BioModels",
    )
    ap.add_argument(
        "--data",
        type=Path,
        default=None,
        help="override the data directory (default: <suite>/data)",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.data is not None:
        global DATA_DIR, SBML_DOWNLOADS_DIR, MODEL_REGISTRY
        DATA_DIR = args.data.resolve()
        SBML_DOWNLOADS_DIR = DATA_DIR / "sbml_downloads"
        MODEL_REGISTRY = DATA_DIR / "model_registry.json"

    # Default: reproducible pinned fetch from the provenance manifest.
    if not args.upstream_latest:
        if not args.manifest.exists():
            logger.error(
                "no provenance manifest at %s -- generate it (generate_provenance_manifest.py) "
                "or pass --upstream-latest to crawl the latest instead.",
                args.manifest,
            )
            return 1
        logger.info("pinned fetch from %s into %s", args.manifest.name, SBML_DOWNLOADS_DIR)
        return 1 if fetch_pinned(args.manifest, SBML_DOWNLOADS_DIR) else 0

    logger.warning(
        "--upstream-latest: crawling BioModels for the LATEST of every model -- NOT reproducible; "
        "sha256 may drift from the pinned manifest."
    )
    ode_only = not args.all_models
    logger.info("fetch target: %s models", "ODE" if ode_only else "ALL")
    logger.info("corpus dir: %s", SBML_DOWNLOADS_DIR)

    available = available_model_ids(ode_only=ode_only, refresh_registry=args.refresh_registry)
    if not available:
        logger.error("no models available from BioModels -- aborting")
        return 1

    on_disk = already_fetched_ids()
    to_fetch = select_models(available, on_disk, n=args.n, strategy=args.strategy)
    if not to_fetch:
        logger.info("corpus is already complete -- nothing to fetch")
        return 0

    logger.info("fetching %d new model(s)...", len(to_fetch))
    from bioservices import BioModels

    bm = BioModels()
    ok, fail = [], []
    for model_id in tqdm(to_fetch, desc="fetching"):
        (ok if download_sbml(bm, model_id) else fail).append(model_id)

    logger.info("fetched %d/%d new models into %s", len(ok), len(to_fetch), SBML_DOWNLOADS_DIR)
    if fail:
        logger.warning(
            "%d failed: %s%s", len(fail), ", ".join(fail[:10]), " ..." if len(fail) > 10 else ""
        )
    logger.info(
        "corpus now holds %d SBML files -- re-run filter.py to refresh manifest.csv",
        len(already_fetched_ids()),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
