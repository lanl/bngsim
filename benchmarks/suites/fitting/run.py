#!/usr/bin/env python3
"""Thin ``suites/fitting/run.py`` driver.

Dispatches to ``harness/run_jobs.py`` (the timing benchmark that
populates the main-text fitting table) and ``harness/consistency_check.py``
(the cross-backend reproducibility gate), so the fitting suite presents
the same ``--mode --effort`` shape every other suite uses.  Both
harness scripts accept ``--effort {low,medium,high}``; this driver
forwards it through.

``run_all.py`` invokes this with ``--mode both --effort <tier>``; users
running the fitting suite standalone get the same UX as every other
``suites/<name>/run.py``.

Usage:
    python run.py                       # both -- consistency + timing
    python run.py --mode correctness    # harness/consistency_check.py only
    python run.py --mode timing         # harness/run_jobs.py only
    python run.py --effort low          # cheap subset (cumulative tiers)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
HARNESS_DIR = BENCH_DIR / "harness"
_BENCH_ROOT = BENCH_DIR.parents[1]  # bngsim/benchmarks/
sys.path.insert(0, str(_BENCH_ROOT))

from _effort import EFFORT_LEVELS  # noqa: E402


def _run_harness(script: str, extra: list[str]) -> int:
    """Run ``harness/<script>`` with ``extra`` argv; return its exit code."""
    cmd = [sys.executable, str(HARNESS_DIR / script), *extra]
    print(f"[fitting] $ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--mode",
        choices=("correctness", "timing", "both"),
        default="both",
        help=(
            "Which gates to run (default: both). 'correctness' runs the "
            "cross-backend consistency check; 'timing' runs the scatter-search "
            "fitting benchmark; 'both' runs each in turn."
        ),
    )
    ap.add_argument(
        "--effort",
        choices=list(EFFORT_LEVELS),
        default="high",
        help=(
            "Cumulative effort tier, forwarded verbatim to each harness "
            "script (default: high -- the full sweep)."
        ),
    )
    # Forward any remaining flags (e.g. --only, --pybnf-cmd, --bngpath) raw.
    args, passthrough = ap.parse_known_args()
    forwarded = ["--effort", args.effort, *passthrough]

    rc = 0
    if args.mode in ("correctness", "both"):
        rc = _run_harness("consistency_check.py", forwarded) or rc
    if args.mode in ("timing", "both"):
        rc = _run_harness("run_jobs.py", forwarded) or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
