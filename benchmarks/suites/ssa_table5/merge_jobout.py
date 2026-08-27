#!/usr/bin/env python3
"""Rebuild results/ssa_timing_ballpark.json from the per-cell results/_jobout/*.json files
(so any re-run cells are picked up) + synthesized N/A cells, re-running aggregate().

Use after re-running individual cells (e.g. the erk/prion COPASI cells under the post-hoc
per-run cap). Does NOT re-run any simulation — pure re-assembly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _ssa_config as C
import run_ssa_timing as R

HERE = Path(__file__).resolve().parent
JOBOUT = HERE / "results" / "_jobout"
JPATH = HERE / "results" / "ssa_timing_ballpark.json"


def build_parser():
    """The CLI: no options, but argv is parsed rather than ignored (GH #489).

    Pure re-assembly -- it re-runs no simulation -- but it does overwrite
    `results/ssa_timing_ballpark.json` from whatever `results/_jobout/`
    currently holds, so a stale or partial `_jobout` becomes the result set.
    """
    return argparse.ArgumentParser(
        prog=Path(__file__).name,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def main(argv=None):
    build_parser().parse_args(argv)
    prev = json.loads(JPATH.read_text()) if JPATH.exists() else {}
    records, missing = [], []
    for k in C.ordered_models():
        for eng in C.ENGINES:
            cov, _ = C.cell_status(k, eng)
            if cov == "na":
                records.append(R.synth_na(k, eng))
                continue
            f = JOBOUT / f"{k}__{eng}.json"
            if f.exists():
                records.append(json.loads(f.read_text()))
            else:
                missing.append((k, eng))
    if missing:
        print("WARNING missing job files:", missing)

    agg = R.aggregate(records)
    payload = dict(prev)
    payload.update(
        {
            "suite": "ssa_table5",
            "engine_set": C.ENGINES,
            "bngsim_version": __import__("bngsim").__version__,
            "per_run_cap_sec": C.PER_RUN_CAP_SEC,
            "cell_wall_cap_sec": C.CELL_WALL_CAP_SEC,
            "warm_budget_sec": C.WARM_BUDGET_SEC,
            "ballpark": True,
            "ballpark_note": "Wall times collected under 6-way concurrency; final results require a clean serial re-run (--workers 1).",
            "reassembled_from_jobout": True,
            "results": sorted(
                records, key=lambda r: (C.MODELS.get(r["model"], {}).get("order", 99), r["engine"])
            ),
            **agg,
        }
    )
    JPATH.write_text(json.dumps(payload, indent=2, default=str))
    print("rebuilt", JPATH, "| status counts:", agg["status_counts"])


if __name__ == "__main__":
    main()
