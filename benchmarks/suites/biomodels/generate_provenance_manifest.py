#!/usr/bin/env python3
"""Generate the version-pinned provenance manifest for the BioModels corpus.

``fetch.py`` downloads each model's *latest* SBML from the BioModels REST API,
and ``manifest.csv`` records the sha256 of the bytes we tested — but neither pins
a *version*, so a re-fetch after BioModels revises a model gives silently
different bytes. This script closes that: for every ``keep`` model it records the
exact upstream version whose sha256 matches our tested bytes, so a re-fetch is
deterministic (``model/download/<id>.<version>`` returns exactly those bytes).

Built mostly from the BioModels **metadata API** (one lightweight call per model,
cached) — no 273 MB corpus download. Each model's metadata carries its revision
history (version numbers) and the latest file's ``sha256sum``:

  * latest sha256sum == our manifest.csv sha256  → pin to the latest version.
  * they differ (our tested bytes are a different OMEX member — e.g. ``_urn.xml``
    or an author-named file — or an older revision) → download the versioned OMEX
    (latest first, then older) and apply fetch.py's member selection until one
    member's sha256 matches our tested bytes; pin that (version, member).
  * no member of any version matches → unresolved drift (the tested bytes then
    live only in the frozen archive tier). The current corpus has zero.

Output conforms to ``../../../parity_checks/manifest.schema.json`` (origin.kind
= ``rest-accession``). This one manifest also covers rr_parity's 332
``biomodels_dir`` SBML (all inside this keep-set).

Usage:
    python generate_provenance_manifest.py                 # full keep-set (crawls)
    python generate_provenance_manifest.py --limit 5       # sample (validate first)
    python generate_provenance_manifest.py --ids BIOMD0000000001,BIOMD0000000002
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST_CSV = HERE / "manifest.csv"
META_CACHE = HERE / "data" / "biomodels_meta_cache.json"  # gitignored (data/)
OMEX_CACHE = HERE / "data" / "biomodels_omex_resolve_cache.json"  # gitignored
DEFAULT_OUT = HERE / "biomodels_provenance.json"

API = "https://www.ebi.ac.uk/biomodels"
DELAY = 0.05  # polite pause between API calls


def read_keep_set() -> dict[str, str]:
    """model_id -> tested sha256, for the manifest.csv ``keep`` rows."""
    keep: dict[str, str] = {}
    with MANIFEST_CSV.open() as f:
        next(f)  # leading "# ..." comment line
        for row in csv.DictReader(f):
            if row["verdict"] == "keep":
                keep[row["model_id"]] = row["sha256"]
    return keep


class Meta:
    """BioModels metadata API with an on-disk cache (gitignored)."""

    def __init__(self) -> None:
        self.cache: dict[str, dict] = {}
        if META_CACHE.exists():
            self.cache = json.loads(META_CACHE.read_text())

    def get(self, mid: str) -> dict:
        if mid not in self.cache:
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(f"{API}/{mid}?format=json", timeout=30) as r:
                        self.cache[mid] = json.load(r)
                    break
                except Exception:  # noqa: BLE001 -- transient network
                    if attempt == 2:
                        raise
                    time.sleep(0.5 * (attempt + 1))
            time.sleep(DELAY)
        return self.cache[mid]

    def flush(self) -> None:
        META_CACHE.parent.mkdir(parents=True, exist_ok=True)
        META_CACHE.write_text(json.dumps(self.cache))


def _looks_sbml(b: bytes) -> bool:
    return bool(b) and b"<sbml" in b[:8192].lower()


def _select_members(names: list[str]) -> list[str]:
    """fetch.py's exact SBML-candidate order within an OMEX archive.

    The tested bytes are whichever member fetch.py picked: prefer ``*_url.xml``,
    then an sbml-named ``.xml``, then any non-manifest ``.xml`` (e.g. ``*_urn.xml``
    or an author-named file). Mirroring it here makes the pin reproduce exactly
    what fetch.py placed on disk."""
    c = [n for n in names if n.endswith("_url.xml")]
    c += [n for n in names if n.endswith(".xml") and "sbml" in n.lower()]
    c += [n for n in names if n.endswith(".xml") and "manifest" not in n.lower()]
    return list(dict.fromkeys(c))


def resolve_via_omex(mid: str, tested_sha: str, latest: int) -> tuple[str, int] | None:
    """Find the (member_filename, version) whose bytes == our tested sha256.

    The lightweight metadata sha256sum only covers the API ``main`` file; some
    models' tested bytes are a *different* OMEX member. Download the versioned
    OMEX (latest first, then older) and apply fetch.py's selection. Returns
    ``(member, version)`` — both re-fetchable via
    ``model/download/<id>.<version>?filename=<member>`` — or None.
    """
    for v in range(latest, 0, -1):
        try:
            with urllib.request.urlopen(f"{API}/model/download/{mid}.{v}", timeout=90) as r:
                zdata = r.read()
            zf = zipfile.ZipFile(io.BytesIO(zdata))
        except Exception:  # noqa: BLE001 -- some versions 403/empty
            continue
        for member in _select_members(zf.namelist()):
            try:
                b = zf.read(member)
            except KeyError:
                continue
            if _looks_sbml(b) and hashlib.sha256(b).hexdigest() == tested_sha:
                return member, v
        time.sleep(DELAY)
    return None


def _license(mid: str) -> dict:
    # Curated BIOMD* are CC0-1.0; non-curated MODEL* are unspecified upstream.
    return {"spdx": "CC0-1.0" if mid.startswith("BIOMD") else "unknown", "notice": None}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--limit", type=int, default=None, help="only the first N keep models (sampling)"
    )
    ap.add_argument("--ids", default=None, help="comma-separated model ids (overrides --limit)")
    ap.add_argument(
        "--snapshot", default=datetime.date.today().isoformat(), help="ISO snapshot date"
    )
    ap.add_argument(
        "--no-resolve", action="store_true", help="flag drift but skip the version walk"
    )
    args = ap.parse_args()

    keep = read_keep_set()
    ids = (
        [s.strip() for s in args.ids.split(",")]
        if args.ids
        else sorted(keep)[: args.limit]
        if args.limit
        else sorted(keep)
    )

    omex_cache: dict[str, list] = json.loads(OMEX_CACHE.read_text()) if OMEX_CACHE.exists() else {}

    meta = Meta()
    records, clean, resolved, unresolved, errors = [], 0, 0, 0, []
    for i, mid in enumerate(ids):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(ids)} ...", file=sys.stderr)
            meta.flush()
        tested = keep[mid]
        try:
            d = meta.get(mid)
        except Exception as exc:  # noqa: BLE001
            errors.append((mid, str(exc)))
            continue
        revs = d.get("history", {}).get("revisions", []) or []
        latest = max((rv.get("version", 0) for rv in revs), default=0)
        main = (d.get("files", {}).get("main") or [{}])[0]
        fname = main.get("name", "")
        api_sha = main.get("sha256sum", "")

        drift = None
        member = fname  # the file to record + re-fetch by name
        if api_sha == tested:
            version, drift = latest, False
            clean += 1
        elif args.no_resolve:
            version, drift = None, "unchecked"
        elif mid in omex_cache:  # previously resolved OMEX member (cached)
            member, version = omex_cache[mid]
            drift = "resolved"
            resolved += 1
        else:
            res = resolve_via_omex(mid, tested, latest)
            if res is not None:
                member, version = res
                omex_cache[mid] = [member, version]
                OMEX_CACHE.write_text(json.dumps(omex_cache))
                drift = "resolved"
                resolved += 1
            else:
                version, drift = None, "unresolved"
                unresolved += 1

        rec = {
            "id": mid,
            "source": "biomodels",
            "origin": {
                "kind": "rest-accession",
                "service": "BioModels",
                "accession": mid,
                "version": str(version) if version is not None else "unresolved",
                "url": f"{API}/{mid}",
                "snapshot": args.snapshot,
            },
            "vendored": None,
            "sha256": tested,
            "license": _license(mid),
            "main_file": member,
            "latest_version": latest,
        }
        if drift:
            rec["drift"] = drift
        records.append(rec)

    meta.flush()

    manifest = {
        "schema_version": "1.0",
        "corpus": "biomodels",
        "generated": args.snapshot,
        "upstream_pin": {"source": "biomodels", "service": "BioModels", "snapshot": args.snapshot},
        "notes": (
            f"{len(records)} keep models pinned to the BioModels version whose sha256 matches our "
            f"tested bytes ({clean} at latest, {resolved} resolved to an older version, "
            f"{unresolved} unresolved drift). Re-fetch deterministically via "
            f"model/download/<id>.<version>. Also covers rr_parity's biomodels_dir SBML."
        ),
        "records": records,
    }
    _write(args.out, manifest)

    print(f"records         : {len(records)}")
    print(f"  clean (latest): {clean}")
    print(f"  drift resolved: {resolved}")
    print(f"  drift UNRESOLV: {unresolved}")
    print(f"errors          : {len(errors)}")
    for mid, e in errors[:10]:
        print(f"  {mid}: {e}", file=sys.stderr)
    print(f"out             : {args.out}")
    return 1 if unresolved or errors else 0


def _write(out: Path, manifest: dict) -> None:
    """Header pretty-printed, one compact record per line (diffable + small)."""
    header = {k: v for k, v in manifest.items() if k != "records"}
    lines = ["{"]
    for k, v in header.items():
        lines.append(f"  {json.dumps(k)}: {json.dumps(v)},")
    lines.append('  "records": [')
    recs = manifest["records"]
    for i, r in enumerate(recs):
        tail = "," if i < len(recs) - 1 else ""
        lines.append("    " + json.dumps(r, separators=(",", ":")) + tail)
    lines.append("  ]")
    lines.append("}")
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
