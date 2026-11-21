"""Fetch a BioModels entry by its id, and keep a copy (issue #1706).

``Model.from_biomodels("BIOMD0000000010")`` asks BioModels' REST API which
file is the entry's main SBML file, downloads it, checks it against the
sha256 BioModels publishes, and keeps it in a cache, so a second call needs
no network. Only the standard library is used, and nothing is fetched unless
a model is asked for by its id.

The cache is ``~/.cache/bngsim/biomodels``, or ``$BNGSIM_BIOMODELS_CACHE_DIR``.
Each file is kept under its sha256, beside a small record per id and version.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from bngsim._exceptions import ModelError

#: BioModels' host. The old https://www.ebi.ac.uk/biomodels/ URLs redirect here.
BASE = "https://www.biomodels.org"

#: A BioModels id: a curated entry (BIOMD) or a non-curated one (MODEL).
ID = re.compile(r"(BIOMD|MODEL)\d{10}\Z")

TIMEOUT = 60.0


def _default_cache_dir() -> Path:
    env = os.environ.get("BNGSIM_BIOMODELS_CACHE_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "bngsim" / "biomodels"


CACHE_DIR = _default_cache_dir()


def _user_agent() -> str:
    from bngsim._version import __version__

    return f"bngsim/{__version__} (+https://github.com/lanl/bngsim)"


def _get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _user_agent()})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        data: bytes = response.read()
        return data


def _record_path(cache_dir: Path, model_id: str, version: int | None) -> Path:
    return cache_dir / f"{model_id}.{'latest' if version is None else version}.json"


def _read_cached(cache_dir: Path, model_id: str, version: int | None) -> dict[str, Any] | None:
    try:
        record: dict[str, Any] = json.loads(_record_path(cache_dir, model_id, version).read_text())
        path = cache_dir / f"{record['sha256']}.xml"
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            return None
    except (OSError, ValueError, KeyError):
        return None
    record["path"] = str(path)
    return record


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _fetch(model_id: str, version: int | None) -> tuple[dict[str, Any], bytes]:
    """The entry's main file: its record (file, version, sha256) and its bytes."""
    entry = model_id if version is None else f"{model_id}.{version}"
    try:
        meta = json.loads(_get(f"{BASE}/{urllib.parse.quote(entry)}?format=json"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ModelError(
                f"BioModels has no entry {entry!r} ({BASE}/{entry} answered 404)"
            ) from None
        raise
    main = (meta.get("files") or {}).get("main") or []
    if not main:
        raise ModelError(f"BioModels lists no main file for {entry!r}")
    name = main[0]["name"]
    published = main[0].get("sha256sum")
    if version is None:
        revisions = (meta.get("history") or {}).get("revisions") or []
        versions = [int(r["version"]) for r in revisions if "version" in r]
        version = max(versions) if versions else None
    where = model_id if version is None else f"{model_id}.{version}"
    data = _get(
        f"{BASE}/model/download/{urllib.parse.quote(where)}?"
        + urllib.parse.urlencode({"filename": name})
    )
    sha256 = hashlib.sha256(data).hexdigest()
    if published and sha256 != published:
        raise ModelError(
            f"the file BioModels sent for {where} ({name}) has sha256 {sha256}, not the "
            f"{published} it publishes for that file; try again later"
        )
    if b"<sbml" not in data[:8192].lower():
        raise ModelError(f"the file BioModels sent for {where} ({name}) is not SBML")
    return {"id": model_id, "version": version, "file": name, "sha256": sha256}, data


def fetch(
    model_id: str, *, version: int | None = None, cache: bool = True, cache_dir: Path | None = None
) -> dict[str, Any]:
    """The entry's main SBML file, from the cache or from BioModels.

    Returns ``{"id", "version", "file", "sha256", "path"}``, where ``path`` is
    the cached file. With ``cache=False`` nothing is read from or written to
    the cache: ``path`` is ``None`` and ``data`` holds the file's bytes.
    """
    if not isinstance(model_id, str) or not ID.match(model_id):
        raise ValueError(
            f"{model_id!r} is not a BioModels id, which looks like 'BIOMD0000000010' "
            "or 'MODEL1006230000'"
        )
    cache_dir = CACHE_DIR if cache_dir is None else Path(cache_dir)
    if cache:
        cached = _read_cached(cache_dir, model_id, version)
        if cached is not None:
            return cached
    try:
        record, data = _fetch(model_id, version)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        cached_note = (
            f"No copy is cached in {cache_dir}."
            if cache
            else "cache=False, so no cached copy was used."
        )
        raise ModelError(
            f"could not fetch {model_id} from {BASE}: {exc}. Without a network, or "
            f"where {BASE} refuses the request (some cloud hosts are refused), "
            f"{cached_note} Download the SBML file yourself and load it with "
            "Model.from_sbml(path)."
        ) from None
    if not cache:
        return {**record, "path": None, "data": data}
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{record['sha256']}.xml"
    _write_atomic(path, data)  # also replaces a damaged copy
    text = json.dumps(record, indent=1).encode()
    _write_atomic(_record_path(cache_dir, model_id, record["version"]), text)
    if version is None:
        _write_atomic(_record_path(cache_dir, model_id, None), text)
    return {**record, "path": str(path)}
