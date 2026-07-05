#!/usr/bin/env python3
"""Build dsmts_index.json from the SBML Test Suite stochastic corpus.

Walks ~/Code/sbml-test-suite/cases/stochastic/ (override with
$SBML_TEST_SUITE_DIR pointed at the parent of `stochastic/`, or
$DSMTS_DIR pointed at `stochastic/` directly), parses each
NNNNN-model.m header for tags/testType/packages, parses
NNNNN-settings.txt for run config, then records SBML levels present
and DSMTS-proper expected-output CSVs.

Output: dsmts_index.json next to this script.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_SUITE_PARENT = Path(
    os.environ.get(
        "SBML_TEST_SUITE_DIR",
        os.path.expanduser("~/Code/sbml-test-suite/cases/semantic"),
    )
).parent  # cases/

DSMTS_DIR = Path(
    os.environ.get(
        "DSMTS_DIR",
        DEFAULT_SUITE_PARENT / "stochastic",
    )
).expanduser()


_HEADER_KEYS = (
    "category",
    "synopsis",
    "componentTags",
    "testTags",
    "testType",
    "levels",
    "generatedBy",
    "packagesPresent",
)


def parse_model_m(path: Path) -> dict:
    text = path.read_text()
    out: dict[str, object] = {}
    # `[^\S\n]*` = horizontal whitespace only; \s would eat the newline and
    # capture the next paragraph for fields with empty values (e.g. packagesPresent).
    for key in _HEADER_KEYS:
        m = re.search(rf"^{key}:[^\S\n]*(.*)$", text, re.MULTILINE)
        out[key] = m.group(1).strip() if m else ""
    return out


def _split_csv(value: str) -> list[str]:
    return [t.strip() for t in value.split(",") if t.strip()]


def _parse_range(value: str) -> list[float] | None:
    m = re.match(r"\(\s*(-?[\d.eE+-]+)\s*,\s*(-?[\d.eE+-]+)\s*\)", value.strip())
    if not m:
        return None
    return [float(m.group(1)), float(m.group(2))]


def parse_settings(path: Path) -> dict:
    raw: dict[str, str] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            k, _, v = line.partition(":")
            raw[k.strip()] = v.strip()
    return {
        "start": float(raw.get("start", "0")),
        "duration": float(raw.get("duration", "1")),
        "steps": int(raw.get("steps", "50")),
        "variables": _split_csv(raw.get("variables", "")),
        "amount": _split_csv(raw.get("amount", "")),
        "concentration": _split_csv(raw.get("concentration", "")),
        "output": _split_csv(raw.get("output", "")),
        "mean_range": _parse_range(raw.get("meanRange", "")),
        "sd_range": _parse_range(raw.get("sdRange", "")),
    }


_SBML_LEVEL_RE = re.compile(r"-sbml-(l\d+v\d+)\.xml$")


def find_sbml_levels(case_dir: Path, case_id: str) -> list[str]:
    levels = []
    for child in case_dir.glob(f"{case_id}-sbml-*.xml"):
        m = _SBML_LEVEL_RE.search(child.name)
        if m:
            levels.append(m.group(1))
    return sorted(levels)


def find_dsmts_csvs(case_dir: Path) -> tuple[str | None, str | None]:
    means = sorted(case_dir.glob("dsmts-*-mean.csv"))
    sds = sorted(case_dir.glob("dsmts-*-sd.csv"))
    mean = means[0].name if means else None
    sd = sds[0].name if sds else None
    return mean, sd


def build_index(suite_root: Path) -> dict:
    if not suite_root.is_dir():
        sys.exit(f"DSMTS dir not found: {suite_root}")

    case_dirs = sorted(p for p in suite_root.iterdir() if p.is_dir() and p.name.isdigit())
    cases: dict[str, dict] = {}
    n_dsmts_proper = 0

    for cd in case_dirs:
        cid = cd.name
        model_m = cd / f"{cid}-model.m"
        settings = cd / f"{cid}-settings.txt"
        results_csv = cd / f"{cid}-results.csv"
        if not model_m.exists() or not settings.exists():
            continue

        header = parse_model_m(model_m)
        settings_parsed = parse_settings(settings)
        levels_present = find_sbml_levels(cd, cid)
        mean_csv, sd_csv = find_dsmts_csvs(cd)
        is_dsmts_proper = header["testType"] == "StochasticTimeCourse"
        if is_dsmts_proper:
            n_dsmts_proper += 1

        def rel(p: Path) -> str:
            return str(p.relative_to(suite_root))

        cases[cid] = {
            "synopsis": header["synopsis"],
            "test_type": header["testType"],
            "component_tags": _split_csv(header["componentTags"]),
            "test_tags": _split_csv(header["testTags"]),
            "packages_present": _split_csv(header["packagesPresent"]),
            "levels_declared": _split_csv(header["levels"]),
            "sbml_levels_present": levels_present,
            "settings": settings_parsed,
            "results_csv": rel(results_csv) if results_csv.exists() else None,
            "dsmts_mean_csv": (rel(cd / mean_csv) if mean_csv else None),
            "dsmts_sd_csv": (rel(cd / sd_csv) if sd_csv else None),
            "is_dsmts_proper": is_dsmts_proper,
        }

    pin = json.loads((HERE.parent / "SUITE_PIN.json").read_text())
    return {
        "_meta": {
            "source": pin["source"],
            "commit": pin["commit"],
            "describe": pin["describe"],
            "n_cases_total": len(cases),
            "n_cases_dsmts_proper": n_dsmts_proper,
            "replicate_guidance": {
                "min": 1000,
                "recommended": 10000,
                "sensitive": 100000,
                "source": "stochastic/README.md",
            },
            "generated": _dt.date.today().isoformat(),
            "notes": [
                "Cases 00001-00039 are DSMTS proper (testType=StochasticTimeCourse) and ship dsmts-NNN-MM-{mean,sd}.csv.",
                "Cases 00040+ are StatisticalDistribution tests that use the SBML 'distrib' package; only NNNNN-results.csv is available.",
                "Suite metadata does not specify a per-case replicate count; use replicate_guidance corpus-wide.",
                "_meta.source/commit record the pinned upstream this index was built from (provenance only; the pin lives in ../SUITE_PIN.json). run_dsmts.py defaults to the in-repo vendored copy under dsmts/cases/ (regenerate via vendor_dsmts_cases.py); the upstream checkout is not required at runtime.",
            ],
        },
        "cases": cases,
    }


def main() -> int:
    index = build_index(DSMTS_DIR)
    out_path = HERE / "dsmts_index.json"
    out_path.write_text(json.dumps(index, indent=2, sort_keys=False) + "\n")
    print(
        f"wrote {out_path} ({index['_meta']['n_cases_total']} cases, "
        f"{index['_meta']['n_cases_dsmts_proper']} DSMTS-proper)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
