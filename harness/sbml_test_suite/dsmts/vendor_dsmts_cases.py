#!/usr/bin/env python3
"""Vendor the DSMTS-proper cases run_dsmts.py needs into the repo.

``run_dsmts.py`` grades BNGsim's SSA against the DSMTS published mean/sd
CSVs. Historically it read those from an *external* SBML Test Suite
checkout (``$SBML_TEST_SUITE_DIR`` / a hardcoded personal path), so the
gate could not run on a bare clone or in CI. This script copies the tiny
subset the loader actually reads into ``cases/`` next to this file so the
gate is self-contained (~140 KB for all 39 cases).

For each ``is_dsmts_proper`` case in ``dsmts_index.json`` it copies, into
``cases/{case_id}/``:

  * ``{case_id}-sbml-{lvl}.xml`` for the *single* level the loader would
    pick (first of ``LEVEL_PREFERENCE`` present in the case's
    ``sbml_levels_present``) — matching ``run_dsmts._resolve_sbml_path``,
  * the ``dsmts-NNN-MM-mean.csv`` / ``-sd.csv`` named in the index,
  * ``{case_id}-settings.txt`` (not read at runtime — settings come from
    the index — but kept so each vendored case is self-describing).

This is a portability fix, NOT a coverage change: the same 39 cases are
copied that ``run_dsmts.py`` already runs. The 61 ``StatisticalDistribution``
cases (SBML ``distrib`` package) are deliberately out of scope — they
cannot be graded by an exact Gillespie SSA via a mean/SD Z-test.

The upstream source is resolved the same way ``run_dsmts.py`` resolves
its (override) suite root:

    --suite-root .../cases/stochastic, else
    $SBML_TEST_SUITE_DIR/cases/stochastic, else
    dsmts_index.json _meta.suite_root

Usage (from anywhere):
    python harness/sbml_test_suite/dsmts/vendor_dsmts_cases.py
    python .../vendor_dsmts_cases.py --suite-root ~/Code/sbml-test-suite/cases/stochastic
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DSMTS_INDEX = HERE / "dsmts_index.json"
VENDOR_DIR = HERE / "cases"

# Keep in sync with run_dsmts.LEVEL_PREFERENCE: the loader picks the first
# of these present in a case's sbml_levels_present, so we vendor exactly
# that one level.
LEVEL_PREFERENCE = ["l3v2", "l3v1", "l2v5", "l2v4", "l2v3", "l2v2", "l2v1"]


def _resolve_upstream_root(index: dict, override: Path | None) -> Path:
    if override is not None:
        root = override.expanduser().resolve()
    else:
        env = os.environ.get("SBML_TEST_SUITE_DIR")
        if env:
            root = (Path(env).expanduser() / "cases" / "stochastic").resolve()
        else:
            root = Path(index["_meta"]["suite_root"]).expanduser().resolve()
    if not root.is_dir():
        sys.exit(
            f"upstream DSMTS suite root not found: {root}\n"
            "Pass --suite-root .../cases/stochastic or set $SBML_TEST_SUITE_DIR."
        )
    return root


def _pick_level(levels: list[str]) -> str | None:
    return next((lvl for lvl in LEVEL_PREFERENCE if lvl in levels), None)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--suite-root",
        type=Path,
        default=None,
        help="Upstream .../cases/stochastic to copy from. Defaults to "
        "$SBML_TEST_SUITE_DIR/cases/stochastic, else dsmts_index.json "
        "_meta.suite_root.",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Remove the existing cases/ dir before vendoring.",
    )
    args = p.parse_args()

    with DSMTS_INDEX.open() as f:
        index = json.load(f)
    upstream = _resolve_upstream_root(index, args.suite_root)

    if args.clean and VENDOR_DIR.exists():
        shutil.rmtree(VENDOR_DIR)
    VENDOR_DIR.mkdir(parents=True, exist_ok=True)

    proper = {cid: c for cid, c in index["cases"].items() if c.get("is_dsmts_proper")}
    total_bytes = 0
    n_files = 0
    for cid, case in sorted(proper.items()):
        lvl = _pick_level(case["sbml_levels_present"])
        if lvl is None:
            sys.exit(
                f"case {cid}: no level in LEVEL_PREFERENCE among {case['sbml_levels_present']}"
            )

        # (src, dst-name) pairs. CSV index entries are paths relative to the
        # upstream root (e.g. "00001/dsmts-001-01-mean.csv"); XML/settings are
        # under {cid}/.
        wanted = [
            (upstream / cid / f"{cid}-sbml-{lvl}.xml", f"{cid}-sbml-{lvl}.xml"),
            (upstream / case["dsmts_mean_csv"], Path(case["dsmts_mean_csv"]).name),
            (upstream / case["dsmts_sd_csv"], Path(case["dsmts_sd_csv"]).name),
            (upstream / cid / f"{cid}-settings.txt", f"{cid}-settings.txt"),
        ]

        dst_dir = VENDOR_DIR / cid
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src, name in wanted:
            if not src.exists():
                sys.exit(f"case {cid}: missing source file {src}")
            dst = dst_dir / name
            shutil.copyfile(src, dst)
            total_bytes += dst.stat().st_size
            n_files += 1

    print(
        f"vendored {len(proper)} DSMTS-proper cases "
        f"({n_files} files, {total_bytes / 1024:.1f} KB) into {VENDOR_DIR}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
