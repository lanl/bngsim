#!/usr/bin/env python3
"""Regenerate the committed unsupported-tags manifest from the SSOT (GH #241).

The manifest — ``bngsim-unsupported-tags.txt`` next to this script — is the list
of SBML Test Suite feature tags bngsim declares it does not support, fed to the
official runner (and to the local :mod:`score` reimplementation) so a refused
case is graded ``Unsupported`` rather than ``Error``. Its single source of truth
is :func:`bngsim._sbml_unsupported.manifest_text`; this script just writes that
text to disk. A unit test (``test_sbml_unsupported_manifest.py``) fails if the
committed file drifts from the SSOT, so run this after changing the SSOT.

    python gen_manifest.py            # rewrite the manifest
    python gen_manifest.py --check    # exit non-zero if it is stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bngsim import _sbml_unsupported

MANIFEST_PATH = Path(__file__).resolve().parent / "bngsim-unsupported-tags.txt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit 1 if the committed manifest is stale.",
    )
    args = parser.parse_args()

    text = _sbml_unsupported.manifest_text()
    if args.check:
        current = MANIFEST_PATH.read_text() if MANIFEST_PATH.exists() else ""
        if current != text:
            sys.stderr.write(
                f"STALE: {MANIFEST_PATH} does not match "
                "bngsim._sbml_unsupported.manifest_text(); run gen_manifest.py\n"
            )
            return 1
        print(f"OK: {MANIFEST_PATH} matches the SSOT.")
        return 0

    MANIFEST_PATH.write_text(text)
    print(f"Wrote {MANIFEST_PATH} ({', '.join(_sbml_unsupported.unsupported_tags())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
