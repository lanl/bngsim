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
import http.client
import json
import numbers
import os
import re
import threading
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
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _fetch(model_id: str, version: int | None) -> tuple[dict[str, Any], bytes]:
    """The entry's main file: its record (file, version, sha256) and its bytes."""
    entry = model_id if version is None else f"{model_id}.{version}"
    try:
        answer = _get(f"{BASE}/{urllib.parse.quote(entry)}?format=json")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ModelError(
                f"BioModels has no entry {entry!r} ({BASE}/{entry} answered 404)"
            ) from None
        raise
    try:
        meta = json.loads(answer)
    except ValueError:
        # A refusal often comes back as a web page with status 200 (#807).
        raise _Unreachable("it answered with a web page, not its API's JSON") from None
    main = (meta.get("files") or {}).get("main") or []
    if not main and version is not None:
        raise ModelError(f"BioModels has no released version {version} of {model_id}")
    if not main:
        raise ModelError(f"BioModels lists no main file for {entry!r}")
    if len(main) > 1:
        # BioModels names an entry's SBML file <id>_url.xml.
        named = [f for f in main if f["name"].endswith("_url.xml")]
        if len(named) != 1:
            names = ", ".join(f["name"] for f in main)
            raise ModelError(
                f"BioModels lists several main files for {entry!r} ({names}); download "
                "the one you want and load it with Model.from_sbml(path)"
            )
        main = named
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


class _Unreachable(Exception):
    """BioModels could not be reached, or answered with something other than its API."""


def fetch(
    model_id: str,
    *,
    version: int | None = None,
    cache: bool = True,
    refresh: bool = False,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """The entry's main SBML file, from the cache or from BioModels.

    Returns ``{"id", "version", "file", "sha256", "path"}``, where ``path`` is
    the cached file. ``refresh=True`` fetches again and updates the cache.
    With ``cache=False`` nothing is read from or written to the cache:
    ``path`` is ``None`` and ``data`` holds the file's bytes.
    """
    if not isinstance(model_id, str) or not ID.match(model_id):
        raise ValueError(
            f"{model_id!r} is not a BioModels id, which looks like 'BIOMD0000000010' "
            "or 'MODEL1006230000'"
        )
    if version is not None and (
        isinstance(version, bool) or not isinstance(version, numbers.Integral) or version < 1
    ):
        raise ValueError(f"version={version!r} is not a version number (1, 2, ...)")
    cache_dir = CACHE_DIR if cache_dir is None else Path(cache_dir)
    if version is not None:
        version = int(version)
    if cache and not refresh:
        cached = _read_cached(cache_dir, model_id, version)
        if cached is not None:
            return cached
    try:
        record, data = _fetch(model_id, version)
    except (_Unreachable, OSError, TimeoutError, http.client.HTTPException) as exc:
        why = exc.reason if isinstance(exc, urllib.error.URLError) else exc
        entry = model_id if version is None else f"{model_id} version {version}"
        if not cache:
            note = f"With cache=False, the cache in {cache_dir} was not read."
        else:
            on_disk = sorted(
                int(v)
                for f in cache_dir.glob(f"{model_id}.*.json")
                if (v := f.name.split(".")[1]).isdigit()
            )
            note = f"No copy of it is cached in {cache_dir}."
            if on_disk:
                note += f" Cached versions: {', '.join(map(str, on_disk))}; pass version=."
        raise ModelError(
            f"could not fetch {entry} from {BASE} ({why}). {note} BioModels refuses "
            "some cloud hosts. You can download the SBML file yourself and load it "
            "with Model.from_sbml(path)."
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
