#!/usr/bin/env python3
"""bngsim wrapper for the official SBML Test Suite test-runner (GH #241).

The SBML Test Suite runner drives a tool through a *wrapper*: for each case it
invokes the wrapper with the argument template ``%d %n %o %l %v`` and then reads
back a CSV the wrapper wrote. This is that wrapper for bngsim.

Argument contract (exactly the runner's substitutions, see
``WrapperConfig.java``)::

    bngsim_wrapper.py  %d  %n  %o  %l  %v
                       │   │   │   │   └ SBML version (int)
                       │   │   │   └──── SBML level (int)
                       │   │   └──────── output directory
                       │   └──────────── case id, e.g. "00042"
                       └──────────────── test-suite cases root (…/cases/semantic)

It reads ``%d/%n/%n-settings.txt`` and ``%d/%n/%n-sbml-l%lv%v.xml``, integrates
with bngsim at the shared suite tolerances, and writes ``%o/%n.csv`` with header
``time,<variables…>`` (the exact ``variables:`` order) and ``steps+1`` rows.

**Fail closed.** The runner's outcome enum keys off whether a CSV exists:

* no CSV + case matches an unsupported tag  ⇒ ``Unsupported`` (honest "can't")
* no CSV + no tag match                     ⇒ ``Error``
* CSV present                               ⇒ ``Match`` / ``NoMatch`` (or
                                              ``CannotSolve`` if tag-excluded)

So on any load refusal (delay / AlgebraicRule), simulation failure, or internal
error, the wrapper writes **no CSV** and exits 0 — never a fabricated or partial
guess. A variable bngsim cannot report is simply omitted from the CSV; the
official grader's ``requireAllColumns`` then turns the gap into a ``NoMatch``,
which is the correct honest outcome (bngsim ran but could not report the value).

The load+integrate path (:func:`build_bngsim_series`) and the column production
(:func:`resolve_columns`) are the *same* shared functions the fair grading
harness uses, so the wrapper cannot be graded through a different code path than
the in-repo ``run.py`` comparison.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

# The shared grading kernel + engine adapters live one directory up.
_SUITE_DIR = Path(__file__).resolve().parent.parent
if str(_SUITE_DIR) not in sys.path:
    sys.path.insert(0, str(_SUITE_DIR))

from _engines import BngsimStageError, build_bngsim_series  # noqa: E402
from _grading import parse_settings, read_sbml_entities, resolve_columns  # noqa: E402


def _fmt(x: float) -> str:
    """Format one value for the delivered CSV.

    Non-finite values are the SBML Test Suite's exact sentinels and must be
    written as the literal tokens ``INF`` / ``-INF`` / ``NaN`` (the official
    ``ResultSet`` parser and ``CompareResultSets.mismatchedBadValues`` key off
    them); finite values use Python's shortest round-tripping repr.
    """
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "INF" if x > 0 else "-INF"
    return repr(float(x))


def _write_csv(out_path: Path, settings: dict, columns: dict) -> None:
    """Write ``time,<variables…>`` with ``steps+1`` rows on the settings grid.

    Only variables bngsim resolved appear (unresolved ones are omitted, never
    fabricated). The time axis is always present so the grader can align rows.
    """
    grid = np.linspace(
        settings["start"],
        settings["start"] + settings["duration"],
        settings["steps"] + 1,
    )
    header = ["time"] + [v for v in settings["variables"] if v in columns]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(grid)):
            row = [_fmt(float(grid[i]))]
            for v in header[1:]:
                row.append(_fmt(float(columns[v][i])))
            w.writerow(row)


def run_case(suite_dir: str, case_id: str, out_dir: str, level: str, version: str) -> int:
    """Produce ``out_dir/case_id.csv`` for one case, or nothing (fail closed)."""
    case_dir = Path(suite_dir) / case_id
    settings_path = case_dir / f"{case_id}-settings.txt"
    sbml_path = case_dir / f"{case_id}-sbml-l{level}v{version}.xml"
    if not settings_path.exists() or not sbml_path.exists():
        return 0  # nothing to run ⇒ no CSV (Unavailable/Unsupported/Error upstream)

    settings = parse_settings(settings_path)
    try:
        series, _ = build_bngsim_series(sbml_path, settings)
        ent = read_sbml_entities(sbml_path)
        columns = resolve_columns(settings, series, ent)
    except BngsimStageError:
        return 0  # load or sim refusal ⇒ fail closed, no CSV
    except Exception:  # noqa: BLE001 — any internal error is also fail-closed
        return 0

    _write_csv(Path(out_dir) / f"{case_id}.csv", settings, columns)
    return 0


def main(argv) -> int:
    if len(argv) < 6:
        sys.stderr.write(
            "usage: bngsim_wrapper.py <suite_dir> <case_id> <out_dir> <level> <version>\n"
        )
        return 2
    return run_case(argv[1], argv[2], argv[3], argv[4], argv[5])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
