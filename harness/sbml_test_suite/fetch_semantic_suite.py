#!/usr/bin/env python3
"""Check out the SBML Test Suite at the pinned commit (semantic corpus).

The semantic test suite (~1824 cases) is an external corpus — too large to
commit — so the harness reads it from ``$SBML_TEST_SUITE_DIR``. To make "reproduce
our Table S8 numbers" deterministic, this clones sbmlteam/sbml-test-suite at the
exact commit recorded in ``SUITE_PIN.json`` (the single source of truth for the
suite version; ``git describe`` there is authoritative, the in-tree VERSION.txt
lags). Point ``$SBML_TEST_SUITE_DIR`` at the result.

    python fetch_semantic_suite.py --dest ~/Code/sbml-test-suite
    export SBML_TEST_SUITE_DIR=~/Code/sbml-test-suite

``--upstream-latest`` clones ``develop`` instead (refresh/extend; unpinned).

The 39 DSMTS-proper cases are already vendored + hashed (dsmts/dsmts_manifest.json)
and need no checkout; this is only for the semantic corpus.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIN_FILE = HERE / "SUITE_PIN.json"


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def main() -> int:
    pin = json.loads(PIN_FILE.read_text())
    repo_url = f"https://github.com/{pin['source']}.git"

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dest", type=Path, default=Path.home() / "Code" / "sbml-test-suite")
    ap.add_argument(
        "--commit", default=pin["commit"], help=f"commit to pin (default {pin['commit'][:12]})"
    )
    ap.add_argument(
        "--upstream-latest",
        action="store_true",
        help="clone develop's tip instead of the pin (refresh/extend; unpinned)",
    )
    args = ap.parse_args()

    if args.dest.exists() and any(args.dest.iterdir()):
        print(f"{args.dest} exists and is not empty; refusing to overwrite.", file=sys.stderr)
        print("Remove it first, or pass --dest elsewhere.", file=sys.stderr)
        return 1

    if args.upstream_latest:
        print(
            "WARNING: --upstream-latest clones develop's tip — NOT the pinned commit.",
            file=sys.stderr,
        )
        r = _run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--depth",
                "1",
                "--branch",
                "develop",
                repo_url,
                str(args.dest),
            ]
        )
        if r.returncode != 0:
            print(f"clone failed:\n{r.stderr}", file=sys.stderr)
            return 1
    else:
        print(f"Cloning {pin['source']} @ {args.commit[:12]} ({pin['describe']})...")
        r = _run(["git", "clone", "--filter=blob:none", "--no-checkout", repo_url, str(args.dest)])
        if r.returncode != 0:
            print(f"clone failed:\n{r.stderr}", file=sys.stderr)
            return 1
        r = _run(["git", "-C", str(args.dest), "checkout", args.commit])
        if r.returncode != 0:
            print(f"checkout of {args.commit} failed:\n{r.stderr}", file=sys.stderr)
            return 1

    head = _run(["git", "-C", str(args.dest), "rev-parse", "HEAD"]).stdout.strip()
    sem = args.dest / "cases" / "semantic"
    n = len([p for p in sem.iterdir() if p.is_dir()]) if sem.is_dir() else 0
    print(f"\n✓ checked out {head[:12]}  ({n} semantic case dirs at {sem})")
    print(f"\nNext: export SBML_TEST_SUITE_DIR={args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
