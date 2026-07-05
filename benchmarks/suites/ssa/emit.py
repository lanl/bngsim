"""
emit.py
-------
Render the SSA benchmark as a Markdown report fragment.

Reads ``results/ssa_results.json`` and writes
``bngsim/benchmarks/reports/generated/ssa.md``.

Table S2: Model | Sp | Rxns | BNGsim (s) | run_net (s) | Speedup.
The row set is the runner's ``MODELS`` registry (12 entries); the
``erk_activation`` row -- ``ssa_skip=True`` because exact SSA is
infeasible on it -- is excluded so the table matches the paper TOC
promise of "10 BNG models".

Cells with no timing yet (``mode=correctness`` runs, or a row that
failed the correctness gate) render as ``\\textit{TBD}``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR.parents[1]))  # bngsim/benchmarks/

from _emit import (  # noqa: E402
    TBD,
    add_audience_arg,
    audience_suffix,
    filter_by_role,
    fmt_speedup,
    fmt_time,
    geomean,
    read_results,
    write_generated,
)

sys.path.insert(0, str(BENCH_DIR))
from run import MODELS  # noqa: E402

DEFAULT_RESULTS = BENCH_DIR / "results" / "ssa_results.json"


def _bn_time(row) -> float | None:
    tim = (row or {}).get("timing", {})
    bng = tim.get("bngsim", {}) if tim else {}
    v = bng.get("median_wall_time")
    return v if (v and v > 0) else None


def _rn_time(row) -> float | None:
    tim = (row or {}).get("timing", {})
    rn = tim.get("run_network", {}) if tim else {}
    v = rn.get("median_wall_time")
    return v if (v and v > 0) else None


def render(payload: dict, models=None) -> str:
    """Render the S2 table.  ``models`` defaults to ``MODELS``; pass a
    filtered subsequence (see :func:`_emit.filter_by_role`) to render a
    per-audience variant."""
    if models is None:
        models = MODELS
    by_name = {r["name"]: r for r in payload.get("results", [])} if payload else {}

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lrrccc}",
        r"\toprule",
        r"Model & Sp & Rxns & BNGsim (s) & run\_net (s) & Speedup \\",
        r"\midrule",
    ]

    speedups: list[float] = []
    for m in models:
        if m.get("ssa_skip"):
            continue
        row = by_name.get(m["name"], {})
        bn_t = _bn_time(row)
        rn_t = _rn_time(row)
        if bn_t and rn_t:
            speedups.append(rn_t / bn_t)

        cells = [
            m["name"].replace("_", r"\_"),
            str(m["species"]),
            str(m["reactions"]),
            fmt_time(bn_t, decimals=4),
            fmt_time(rn_t, decimals=4),
            fmt_speedup(rn_t, bn_t),
        ]
        lines.append(" & ".join(cells) + r" \\")

    gm = geomean(speedups)
    lines.append(r"\midrule")
    geo_cell = f"{gm:.1f}$\\times$" if gm else TBD
    lines.append(
        r"\multicolumn{5}{l}{\textbf{Geometric mean speedup:}} "
        + r"& \textbf{"
        + geo_cell
        + r"} \\"
    )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-in", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--dry-run", action="store_true")
    add_audience_arg(parser)
    args = parser.parse_args()

    payload = read_results(args.results_in)
    if not payload:
        print(f"[emit] no results at {args.results_in}; rendering TBD rows.")
    models = filter_by_role(MODELS, args.audience)
    body = render(payload if isinstance(payload, dict) else {}, models=models)
    if args.dry_run:
        print(body)
        return 0
    out = write_generated("ssa" + audience_suffix(args.audience), body)
    print(f"[emit] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
