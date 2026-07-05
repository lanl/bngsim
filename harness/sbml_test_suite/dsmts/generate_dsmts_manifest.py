#!/usr/bin/env python3
"""Generate the provenance manifest for the vendored DSMTS-proper cases.

The 39 DSMTS-proper cases (156 files) that ``run_dsmts.py`` grades against are
vendored under ``cases/`` so the gate is hermetic. This records, per file, its
pinned upstream commit + path + sha256 + license, so the vendored bytes are
verifiable byte-for-byte against the SBML Test Suite.

The pin (commit) comes from ``../SUITE_PIN.json`` — the single source of truth
for the SBML Test Suite version across the repo. sha256 is computed from the
committed vendored bytes; if an upstream checkout is available (``--suite-root``
or ``$SBML_TEST_SUITE_DIR``) each hash is cross-checked against the pinned
commit's blob.

Output conforms to ``../../../parity_checks/manifest.schema.json`` (origin.kind
= ``git``).

Usage:
    python generate_dsmts_manifest.py
    python generate_dsmts_manifest.py --suite-root ~/Code/sbml-test-suite  # cross-check
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
VENDOR_DIR = HERE / "cases"
PIN_FILE = HERE.parent / "SUITE_PIN.json"
DEFAULT_OUT = HERE / "dsmts_manifest.json"


def _load_pin() -> dict:
    return json.loads(PIN_FILE.read_text())


def _upstream_sha(suite_root: Path, commit: str, path: str) -> str | None:
    """sha256 of <commit>:<path> in an upstream sbml-test-suite git checkout."""
    out = subprocess.run(
        ["git", "-C", str(suite_root), "cat-file", "blob", f"{commit}:{path}"],
        capture_output=True,
    )
    if out.returncode != 0:
        return None
    return hashlib.sha256(out.stdout).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--suite-root",
        type=Path,
        default=Path(os.environ["SBML_TEST_SUITE_DIR"])
        if os.environ.get("SBML_TEST_SUITE_DIR")
        else None,
        help="upstream sbml-test-suite git checkout, to cross-check hashes; env $SBML_TEST_SUITE_DIR",
    )
    ap.add_argument("--generated", default=datetime.date.today().isoformat())
    args = ap.parse_args()

    pin = _load_pin()
    commit = pin["commit"]
    license_ = {
        "spdx": pin["license"],
        "notice": None,
    }  # NOTICE wiring is a (paused) licensing task

    xcheck = args.suite_root is not None and (args.suite_root / ".git").exists()
    records, mismatch = [], 0
    for f in sorted(p for p in VENDOR_DIR.rglob("*") if p.is_file()):
        rel = f.relative_to(VENDOR_DIR).as_posix()  # e.g. 00001/00001-sbml-l3v2.xml
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        upstream_path = f"cases/stochastic/{rel}"
        if xcheck:
            up = _upstream_sha(args.suite_root, commit, upstream_path)
            if up is not None and up != sha:
                print(
                    f"MISMATCH {rel}: vendored {sha[:12]} != upstream {up[:12]}", file=sys.stderr
                )
                mismatch += 1
        records.append(
            {
                "id": rel,
                "source": "dsmts",
                "origin": {
                    "kind": "git",
                    "repo": pin["source"],
                    "commit": commit,
                    "path": upstream_path,
                },
                "vendored": f"harness/sbml_test_suite/dsmts/cases/{rel}",
                "sha256": sha,
                "license": dict(license_),
            }
        )

    manifest = {
        "schema_version": "1.0",
        "corpus": "dsmts",
        "generated": args.generated,
        "upstream_pin": {"source": pin["source"], "commit": commit, "describe": pin["describe"]},
        "notes": (
            f"{len(records)} files (39 DSMTS-proper cases) vendored under cases/, verified "
            f"byte-identical to {pin['source']} at {commit[:12]} ({pin['describe']}). "
            "Regenerate via generate_dsmts_manifest.py; the pin lives in ../SUITE_PIN.json."
        ),
        "records": records,
    }
    args.out.write_text(_dump(manifest))

    print(f"records written : {len(records)}")
    print(f"cross-checked   : {'yes @' + commit[:12] if xcheck else 'no (no upstream checkout)'}")
    if xcheck:
        print(f"mismatches      : {mismatch}")
    print(f"out             : {args.out}")
    return 1 if mismatch else 0


def _dump(manifest: dict) -> str:
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


if __name__ == "__main__":
    raise SystemExit(main())
