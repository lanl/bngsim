#!/usr/bin/env python3
"""Promote an SSA-screen run to the committed baseline (GH #108).

Reads a NATIVE ``ssa_screen.json`` payload — a fresh ``runs/ssa_screen.json``, or
the legacy ``harness/results/rr_ssa_biomodels_parity.json`` for the initial seed —
and writes ``ssa_baseline.json``: the same run re-expressed in the shared ``_core``
report schema via :mod:`ssa_attribution`. The regression diff (GH #108 step 4)
compares a fresh run against this baseline.

Re-baseline DELIBERATELY — only when a change in per-model outcomes has been
reviewed and accepted — never on every run. The canonical run config the baseline
should be produced at is pinned in ``ssa_spec.json``.

Usage:
    python make_ssa_baseline.py [--from runs/ssa_screen.json] [--out ssa_baseline.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # so `from _core import ...` resolves
sys.path.insert(0, str(HERE))  # so `import ssa_attribution` resolves
import ssa_attribution as sa  # noqa: E402
from _core import write_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--from",
        dest="src",
        type=Path,
        default=HERE / "runs" / "ssa_screen.json",
        help="Native screen JSON to promote (default runs/ssa_screen.json).",
    )
    ap.add_argument("--out", type=Path, default=HERE / "ssa_baseline.json")
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"missing screen payload: {args.src}")

    payload = json.loads(args.src.read_text())
    meta, results = sa.payload_to_report(payload)
    meta["baseline_source"] = args.src.name
    write_report(args.out, results, meta=meta)
    print(
        f"wrote {args.out}: {len(results)} results  "
        f"tally={meta['outcome_tally']}  "
        f"subclasses={meta['subclass_breakdown']}  "
        f"not_expected={meta['n_not_expected']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
