#!/usr/bin/env python3
"""Batch exporter: bngsim → an official SBML Test Suite results ZIP (GH #241).

Runs bngsim over the whole semantic suite and writes one ``<case>.csv`` per case
it can solve, then zips them into the archive the online SBML Test Suite Database
(https://sbml.bioquant.uni-heidelberg.de) accepts on its *Submit results* form.

Per the database's stated format: each result CSV's filename must **contain the
5-digit case number** (bngsim writes bare ``NNNNN.csv``, which qualifies), and the
upload is a **ZIP** of those files; **gaps are allowed** (a case you don't run or
can't simulate is simply absent). Each CSV is the exact runner format — header
``time,<variables…>`` in the settings ``variables:`` order, ``steps+1`` rows on
the settings grid, amounts/concentrations per settings, ``INF``/``-INF``/``NaN``
sentinels.

This is a **thin batch loop over** :func:`bngsim_wrapper.run_case` — the SAME
per-case path the official runner drives and :mod:`score.py` grades — so the
numbers are byte-identical to both. **Fail-closed:** a case bngsim cannot
faithfully simulate (DAE / DDE / fast / fbc, or any error) gets no CSV, never a
guess; declare the unsupported tags (printed at the end) on the form so those
score ``Unsupported`` rather than ``Error``. Non-TimeCourse cases are skipped up
front. Each case is exported at its **highest available SBML level/version** (the
runner's "Highest" policy, matching :mod:`score.py`); pin one with
``--level/--version``.

Usage::

    python export_results.py --out-dir ~/bngsim-sbml-results
    python export_results.py --out-dir OUT --quick 100
    python export_results.py --out-dir OUT --case 00042

Dev-only (the wheel packages only ``python/bngsim``). Run ``score.py`` for the
Match/NoMatch/Unsupported breakdown before submitting.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

# Shared kernel + engine adapter live one directory up; the effort helper two
# more up (bngsim/benchmarks). Same path wiring as score.py / bngsim_wrapper.py.
_THIS_DIR = Path(__file__).resolve().parent
_SUITE_DIR = _THIS_DIR.parent
_BENCH_ROOT = _SUITE_DIR.parents[1]  # bngsim/benchmarks
for _p in (str(_THIS_DIR), str(_SUITE_DIR), str(_BENCH_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bngsim  # noqa: E402  — for the version string in the submission guidance
from _effort import add_effort_arg, effort_allows  # noqa: E402
from bngsim import _sbml_unsupported  # noqa: E402  — canonical unsupported-tag parser
from bngsim_wrapper import run_case  # noqa: E402  — the vetted per-case export path
from score import (  # noqa: E402  — reuse the vetted suite-walk helpers
    SBML_PREF,
    effort_tier,
    find_sbml_file,
    parse_model_desc,
)

DEFAULT_SUITE_DIR = Path(
    os.environ.get(
        "SBML_TEST_SUITE_DIR",
        os.path.expanduser("~/Code/sbml-test-suite/cases/semantic"),
    )
)
DEFAULT_MANIFEST = _THIS_DIR / "bngsim-unsupported-tags.txt"

_LEVEL_RE = re.compile(r"-sbml-l(\d)v(\d)\.xml$")


def _pick_level_version(case_dir: Path, case_id: str, level: str | None, version: str | None):
    """Return ``(level, version)`` for the SBML file to export, or ``None`` if
    absent. Pinned ``--level/--version`` require exactly that file; otherwise the
    highest available (the runner's "Highest supported" policy)."""
    if level and version:
        if (case_dir / f"{case_id}-sbml-l{level}v{version}.xml").exists():
            return level, version
        return None
    sbml_path = find_sbml_file(case_dir, case_id)
    if sbml_path is None:
        return None
    m = _LEVEL_RE.search(sbml_path.name)
    return (m.group(1), m.group(2)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export bngsim's SBML Test Suite results ZIP for database submission"
    )
    ap.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    ap.add_argument(
        "--out-dir", type=Path, required=True, help="Directory to write <case>.csv into"
    )
    ap.add_argument("--case", type=str, default="", help="Export a single case (e.g. 00001)")
    ap.add_argument("--quick", type=int, default=0, help="First N cases after filters")
    ap.add_argument(
        "--level",
        type=str,
        default=None,
        help="Pin SBML level (e.g. 3); default: highest per case",
    )
    ap.add_argument(
        "--version", type=str, default=None, help="Pin SBML version (e.g. 2); requires --level"
    )
    ap.add_argument(
        "--no-zip", action="store_true", help="Write the CSVs but skip building the upload ZIP"
    )
    add_effort_arg(ap)
    args = ap.parse_args()

    if bool(args.level) != bool(args.version):
        print("ERROR: --level and --version must be given together.")
        return 2
    if not args.suite_dir.exists():
        print(f"ERROR: SBML Test Suite not found at {args.suite_dir}")
        print(
            "  Set SBML_TEST_SUITE_DIR or pass --suite-dir "
            "(fetch with harness/sbml_test_suite/fetch_semantic_suite.py)."
        )
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.case:
        case_dirs = [args.suite_dir / args.case]
    else:
        case_dirs = sorted(d for d in args.suite_dir.iterdir() if d.is_dir() and d.name.isdigit())
        case_dirs = [d for d in case_dirs if effort_allows(args.effort, effort_tier(d.name))]
        if args.quick > 0:
            case_dirs = case_dirs[: args.quick]

    pinned = f"l{args.level}v{args.version}" if args.level else "highest-per-case"
    print(f"Exporting {len(case_dirs)} case(s) at {pinned} → {args.out_dir}")

    written, no_file, not_timecourse, unsolved = [], [], [], []
    t0 = time.time()
    for i, case_dir in enumerate(case_dirs):
        case_id = case_dir.name

        # bngsim is a TimeCourse ODE engine; non-TimeCourse types (fbc steady
        # state, etc.) are out of scope → no CSV (declared-Unsupported on the DB).
        if parse_model_desc(case_dir / f"{case_id}-model.m")["testType"] != "TimeCourse":
            not_timecourse.append(case_id)
            continue

        lv = _pick_level_version(case_dir, case_id, args.level, args.version)
        if lv is None:
            no_file.append(case_id)
            continue

        # The vetted per-case export path: writes out_dir/<case_id>.csv, or
        # nothing (fail-closed) on any refusal/error. Never raises.
        run_case(str(args.suite_dir), case_id, str(args.out_dir), lv[0], lv[1])
        (written if (args.out_dir / f"{case_id}.csv").exists() else unsolved).append(case_id)

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(case_dirs)}  ({time.time() - t0:.0f}s, written={len(written)})")

    # Declared unsupported tags — comma-joined, paste-ready for the form field.
    tags = (
        _sbml_unsupported.parse_manifest(DEFAULT_MANIFEST.read_text())
        if DEFAULT_MANIFEST.exists()
        else []
    )
    tags_line = ", ".join(tags)

    (args.out_dir / "_export_summary.json").write_text(
        json.dumps(
            {
                "bngsim_version": bngsim.__version__,
                "suite_dir": str(args.suite_dir),
                "level_version": pinned,
                "counts": {
                    "considered": len(case_dirs),
                    "written": len(written),
                    "fail_closed": len(unsolved),
                    "non_timecourse": len(not_timecourse),
                    "no_sbml": len(no_file),
                },
                "written": written,
                "fail_closed": unsolved,
                "non_timecourse": not_timecourse,
                "no_sbml": no_file,
                "unsupported_tags": tags,
                "sbml_pref": SBML_PREF,
            },
            indent=2,
        )
    )

    # Build the uploadable ZIP: ONLY the <case>.csv files (flat archive). The
    # summary json carries no case number and is deliberately left out so the
    # database's filename→case-number extraction sees only real results.
    zip_path = args.out_dir.parent / f"{args.out_dir.name}.zip"
    if not args.no_zip:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for csv in sorted(args.out_dir.glob("*.csv")):
                z.write(csv, arcname=csv.name)

    print(f"\n{'=' * 64}")
    print("bngsim → SBML Test Suite results bundle")
    print(f"{'=' * 64}")
    print(f"  cases considered:      {len(case_dirs)}")
    print(f"  result CSVs written:   {len(written)}")
    print(f"  fail-closed (no CSV):  {len(unsolved)}  (unsupported construct / error)")
    print(f"  non-TimeCourse:        {len(not_timecourse)}  (out of scope)")
    print(f"  no SBML at level:      {len(no_file)}")

    print(
        "\n  Submit at https://sbml.bioquant.uni-heidelberg.de  (Register → Login → Submit results):"
    )
    if not args.no_zip:
        print(f"    • ZIP archive:        {zip_path}   ← upload this ({len(written)} CSVs)")
    print(f"    • Description:        bngsim {bngsim.__version__} — ODE, highest SBML level")
    print(f"    • UnsupportedTags:    {tags_line}")
    print("    • SBML Level/Version: Highest")
    print(
        f'    • Simulator:          Create New Simulator → "bngsim" (version {bngsim.__version__})'
    )
    print(
        f"\n  Run score.py first for the Match/NoMatch/Unsupported breakdown.  [{time.time() - t0:.0f}s]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
