#!/usr/bin/env python3
"""Convert BioModels SBML .xml files to Antimony .ant format.

Uses libAntimony to load SBML and emit canonical Antimony.

Default mode (**BioModels Antimony pool**): ensures the **full manifest union** (~1013 models:
``all_required_ids``) under ``biomodels_ant/``. SBML is fetched into
``biomodels_ssa/data/sbml_downloads/`` when needed.
See ``biomodels_ssa/biomodels_ant_pool.py`` and ``biomodels_ssa/biomodels_ant_pool_manifest.json``.

Legacy mode: batch-convert ``*_original.xml`` from a directory you supply via ``SBML_VALIDATED_SRC``.

Usage:
    python convert_sbml_to_ant.py                     # ensure pool + convert gaps
    python convert_sbml_to_ant.py --no-ensure-pool    # skip automatic fetch/ensure
    SBML_VALIDATED_SRC=/path/to/xml/dir python convert_sbml_to_ant.py --legacy
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from glob import glob
from pathlib import Path

_BENCH = Path(__file__).resolve().parent
_BIOMODELS = _BENCH / "biomodels_ssa"
if str(_BIOMODELS) not in sys.path:
    sys.path.insert(0, str(_BIOMODELS))

from biomodels_ant_pool import (  # noqa: E402
    default_ant_output_dir,
    ensure_biomodels_ant_pool,
    manifest_path,
    pool_ensure_succeeded,
    pool_readiness,
)


def legacy_main(src: str, dst: str) -> int:
    import antimony as ant

    os.makedirs(dst, exist_ok=True)

    xml_files = sorted(glob(os.path.join(src, "*_original.xml")))
    print(f"Found {len(xml_files)} SBML files")

    ok = 0
    fail = 0
    for xf in xml_files:
        name = Path(xf).stem.replace("_original", "")

        ant.clearPreviousLoads()
        ret = ant.loadSBMLFile(str(xf))
        if ret == -1:
            err = ant.getLastError()
            print(f"  FAIL {name}: {err[:60]}")
            fail += 1
            continue

        mod = ant.getModuleNames()[-1]
        ant_str = ant.getAntimonyString(mod)
        if not ant_str:
            print(f"  FAIL {name}: empty antimony string")
            fail += 1
            continue

        out_path = os.path.join(dst, name + ".ant")
        with open(out_path, "w") as f:
            f.write(ant_str)
        ok += 1

    print(f"\nConverted: {ok}/{ok + fail}")
    print(f"Failed: {fail}")
    print(f"Output: {dst}")
    return 0 if fail == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Batch mode: convert *_original.xml from SRC (ssys layout)",
    )
    parser.add_argument(
        "--ensure-pool",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Before converting, fetch SBML / write .ant for biomodels_ant_pool manifest (default: on)",
    )
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="With default mode: only convert when SBML already exists under biomodels_ssa/data/sbml_downloads",
    )
    parser.add_argument(
        "--ant-dir",
        type=Path,
        default=None,
        help="Output directory for .ant files (default: benchmarks/biomodels_ant)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.legacy:
        src = os.environ.get("SBML_VALIDATED_SRC", "").strip()
        if not src:
            print(
                "ERROR: legacy mode requires SBML_VALIDATED_SRC pointing at a directory of "
                "*_original.xml files.",
                file=sys.stderr,
            )
            return 1
        src = os.path.expanduser(src)
        dst = os.path.join(os.path.dirname(__file__), "biomodels_ant")
        return legacy_main(src, dst)

    ant_dir = args.ant_dir or default_ant_output_dir()

    if not manifest_path().is_file():
        print(
            f"ERROR: missing manifest {manifest_path()}.\n"
            "See bngsim/benchmarks/biomodels_ant/README.txt.",
            file=sys.stderr,
        )
        return 1

    if args.ensure_pool:
        summary = ensure_biomodels_ant_pool(
            ant_dir,
            fetch_network=not args.no_fetch,
            include_review_extras=True,
        )
        logging.info(
            "Pool ensure: status=%s present=%s/%s fetch_failed=%s convert_failed=%s",
            summary.get("status"),
            summary.get("present_after", summary.get("already_present")),
            summary.get("required_count"),
            len(summary.get("fetch_failed") or []),
            len(summary.get("convert_failed") or []),
        )
        if summary.get("error"):
            logging.error("%s", summary["error"])
        if summary.get("fetch_failed"):
            logging.warning("Fetch failures (first 10): %s", (summary["fetch_failed"])[:10])
        if summary.get("convert_failed"):
            logging.warning("Conversion failures (first 10): %s", (summary["convert_failed"])[:10])

        if not pool_ensure_succeeded(summary):
            if summary.get("status") == "incomplete_missing_sbml" and args.no_fetch:
                print(
                    "ERROR: missing SBML for some IDs; run without --no-fetch to download.",
                    file=sys.stderr,
                )
            elif summary.get("status") == "incomplete_no_antimony":
                print(
                    "ERROR: install antimony to convert SBML → .ant (see pool summary hint).",
                    file=sys.stderr,
                )
            else:
                print(
                    f"ERROR: pool ensure did not complete (status={summary.get('status')!r}). "
                    "Fix network, install dependencies, or run with SBML pre-cached.",
                    file=sys.stderr,
                )
            return 1

    missing_after, present = pool_readiness(
        ant_dir,
        include_review_extras=True,
    )
    if missing_after:
        logging.warning(
            "%d .ant file(s) still missing: %s …", len(missing_after), missing_after[:5]
        )
        return 1

    print(f"BioModels Antimony pool ready: {len(present)} .ant file(s) under {ant_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
