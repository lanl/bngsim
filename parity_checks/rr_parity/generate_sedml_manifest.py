#!/usr/bin/env python3
"""Generate the pinned provenance manifest for the temp-biomodels artifacts.

The rr_parity ODE corpus pulls two kinds of file from sys-bio/temp-biomodels
(PMC12677764): the SED-ML simulation protocols and, for the ~991 jobs whose
``sbml_origin == 'temp-biomodels'``, the SBML the SED-ML was authored against.
Both are gitignored and re-fetched by ``fetch_sedml.py``; until now they were
pulled from ``@main`` with no pin, so a re-fetch after upstream moved gave a
silently different corpus.

This script writes a manifest (schema: ``../manifest.schema.json``)
recording, for every temp-biomodels file the jobs reference, its pinned upstream
commit + path + sha256 + license, so a re-fetch is verifiable byte-for-byte.

**Scope.** Only the *fetched* temp-biomodels artifacts. The 331 ``invented``
protocols (``horizon_source == 'invented'``) are authored by us and live inline
in the committed ``ode_jobs.json`` — they have no upstream and no file to hash,
so they are out of scope here (recorded in the manifest header for completeness).
The 332 ``biomodels_dir`` SBML come from BioModels REST and are a separate pin.

Sources (env-resolved; never hardcode paths):
    $BIOMODELS_TEMP_REPO   a git checkout of sys-bio/temp-biomodels (authoritative
                           mode: commit read from HEAD, bytes hashed via git blobs)
      -- or --
    $BIOMODELS_SEDML_DIR   the materialized ``final/`` copy (fallback mode: plain
                           files hashed, commit trusted from --commit)

Usage:
    BIOMODELS_TEMP_REPO=/path/to/temp-biomodels python generate_sedml_manifest.py
    python generate_sedml_manifest.py --repo-dir DIR --out temp_biomodels_manifest.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PINNED_COMMIT = "6a09daf46af1bb89e4857436b623a7b8720863ad"
REPO = "sys-bio/temp-biomodels"
LICENSE = {"spdx": "CC0-1.0", "notice": None}  # temp-biomodels ships a CC0-1.0 LICENSE


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


def _sha256_git_blob(repo: Path, commit: str, path: str) -> str | None:
    """sha256 of the file's bytes at <commit>:<path>, fetched lazily if blob-filtered."""
    out = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "blob", f"{commit}:{path}"],
        capture_output=True,
    )
    if out.returncode != 0:
        return None
    return hashlib.sha256(out.stdout).hexdigest()


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump(manifest: dict) -> str:
    """Serialize with the header pretty-printed and one compact record per line.

    Valid JSON, but line-diffable AND compact (~2000 records blow past the
    1 MB large-file guard when pretty-printed with indent=2).
    """
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
    return "\n".join(lines) + "\n"


def referenced_files(jobs: list[dict]) -> list[tuple[str, str, str, str]]:
    """Every temp-biomodels file the jobs depend on.

    Returns (relpath_under_final, model_id, artifact, horizon_source) tuples,
    deduped and sorted. ``relpath_under_final`` doubles as the manifest ``id``.
    """
    seen: dict[str, tuple[str, str, str, str]] = {}
    for j in jobs:
        p = j.get("params", {})
        mid = j["model_id"]
        # SED-ML protocol (figure_sedml / template_sedml tiers).
        sedml = p.get("sedml")
        if sedml:
            rel = f"{mid}/{sedml}"
            seen[rel] = (rel, mid, "sedml", p.get("horizon_source", ""))
        # SBML, only when it came from temp-biomodels (not the biomodels_dir REST set).
        if p.get("sbml_origin") == "temp-biomodels":
            sbml_name = Path(j["model"]).name
            rel = f"{mid}/{sbml_name}"
            seen[rel] = (rel, mid, "sbml", p.get("horizon_source", ""))
    return sorted(seen.values())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--jobs", type=Path, default=HERE / "ode_jobs.json")
    ap.add_argument("--out", type=Path, default=HERE / "temp_biomodels_manifest.json")
    ap.add_argument(
        "--commit",
        default=PINNED_COMMIT,
        help=f"upstream commit to pin (default {PINNED_COMMIT[:12]})",
    )
    ap.add_argument(
        "--repo-dir",
        type=Path,
        default=Path(os.environ["BIOMODELS_TEMP_REPO"])
        if os.environ.get("BIOMODELS_TEMP_REPO")
        else None,
        help="git checkout of sys-bio/temp-biomodels (authoritative mode); env $BIOMODELS_TEMP_REPO",
    )
    ap.add_argument(
        "--source-dir",
        type=Path,
        default=Path(os.environ["BIOMODELS_SEDML_DIR"])
        if os.environ.get("BIOMODELS_SEDML_DIR")
        else None,
        help="materialized final/ copy (fallback mode); env $BIOMODELS_SEDML_DIR",
    )
    ap.add_argument(
        "--generated",
        default=datetime.date.today().isoformat(),
        help="ISO date stamp (default: today)",
    )
    args = ap.parse_args()

    jobs = json.loads(args.jobs.read_text())["jobs"]
    files = referenced_files(jobs)

    # Resolve hashing mode.
    repo = args.repo_dir
    git_mode = repo is not None and (repo / ".git").exists()
    if git_mode:
        head = _git(repo, "rev-parse", "HEAD").strip()
        if head != args.commit:
            print(
                f"WARNING: $BIOMODELS_TEMP_REPO HEAD {head[:12]} != pinned {args.commit[:12]}; "
                f"hashing at the pinned commit regardless.",
                file=sys.stderr,
            )
    elif args.source_dir is None:
        print(
            "ERROR: need either a git checkout ($BIOMODELS_TEMP_REPO / --repo-dir) or a "
            "materialized copy ($BIOMODELS_SEDML_DIR / --source-dir) to hash.",
            file=sys.stderr,
        )
        return 2

    records = []
    missing = []
    for rel, mid, artifact, hsrc in files:
        path_in_repo = f"final/{rel}"
        if git_mode:
            sha = _sha256_git_blob(repo, args.commit, path_in_repo)
        else:
            sha = _sha256_file(args.source_dir / rel)
        if sha is None:
            missing.append(rel)
            continue
        rec = {
            "id": rel,
            "source": "temp-biomodels",
            "origin": {"kind": "git", "repo": REPO, "commit": args.commit, "path": path_in_repo},
            "vendored": None,
            "sha256": sha,
            "license": dict(LICENSE),
            "model_id": mid,
            "artifact": artifact,
        }
        if artifact == "sedml":
            rec["horizon_source"] = hsrc
        records.append(rec)

    n_invented = sum(1 for j in jobs if j.get("params", {}).get("horizon_source") == "invented")
    n_sedml = sum(1 for r in records if r["artifact"] == "sedml")
    n_sbml = sum(1 for r in records if r["artifact"] == "sbml")

    manifest = {
        "schema_version": "1.0",
        "corpus": "rr_parity/temp-biomodels",
        "generated": args.generated,
        "upstream_pin": {"source": "temp-biomodels", "repo": REPO, "commit": args.commit},
        "notes": (
            f"{n_sedml} SED-ML + {n_sbml} SBML fetched from {REPO} at the pinned commit. "
            f"The {n_invented} 'invented' protocols (horizon_source=invented) are authored "
            "in ode_jobs.json and have no upstream file; the biomodels_dir SBML are pinned "
            "separately (BioModels REST)."
        ),
        "records": records,
    }
    args.out.write_text(_dump(manifest))

    print(f"records written : {len(records)}  ({n_sedml} SED-ML + {n_sbml} SBML)")
    print(f"invented (skip) : {n_invented}  (authored in ode_jobs.json)")
    print(
        f"mode            : {'git blobs @ ' + args.commit[:12] if git_mode else 'plain files @ ' + str(args.source_dir)}"
    )
    print(f"out             : {args.out}")
    if missing:
        print(f"MISSING ({len(missing)}) — not found in source:", file=sys.stderr)
        for m in missing[:20]:
            print(f"  {m}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
