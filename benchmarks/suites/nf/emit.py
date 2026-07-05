"""
emit.py
-------
Render the Network-free benchmark as a Markdown report fragment.

Reads ``results/nf_sweep_results.json`` (a flat list, no
``machine_info`` wrapper -- unlike the other ``_netbench``-style
runners) and writes ``bngsim/benchmarks/reports/generated/nf.md``.

Table S4: Model | $t_\\mathrm{end}$ | BNGsim (s) | BNG2.pl (s) |
Speedup.  The row set is the runner's ``CORE_MODELS`` registry (the
default set; the experimental canaries are opt-in via env var and
stay out of the paper artifact).  Cells with no timing render as
``\\textit{TBD}``.
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
    filter_items_by_role,
    fmt_speedup,
    fmt_time,
    geomean,
    read_results,
    write_generated,
)

sys.path.insert(0, str(BENCH_DIR))
from run import CORE_MODELS  # noqa: E402

DEFAULT_RESULTS = BENCH_DIR / "results" / "nf_sweep_results.json"


def render(payload, items=None) -> str:
    """Render the S4 table.  ``items`` defaults to ``CORE_MODELS.items()``;
    pass a filtered subsequence (see :func:`_emit.filter_items_by_role`)
    to render a per-audience variant."""
    if items is None:
        items = list(CORE_MODELS.items())
    # nf runner emits a flat list, not a {results: [...]} wrapper.
    rows = payload if isinstance(payload, list) else payload.get("results", [])
    by_name = {r["model"]: r for r in rows if isinstance(r, dict) and "model" in r}

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Model & $t_\mathrm{end}$ & BNGsim (s) & BNG2.pl (s) & Speedup \\",
        r"\midrule",
    ]

    speedups: list[float] = []
    for name, spec in items:
        row = by_name.get(name, {})
        bn_t = row.get("bngsim_time")
        rn_t = row.get("bng_time")
        bn_t = bn_t if (bn_t and bn_t > 0) else None
        rn_t = rn_t if (rn_t and rn_t > 0) else None
        if bn_t and rn_t:
            speedups.append(rn_t / bn_t)
        cells = [
            name.replace("_", r"\_"),
            str(spec["t_end"]),
            fmt_time(bn_t, decimals=3),
            fmt_time(rn_t, decimals=3),
            fmt_speedup(rn_t, bn_t),
        ]
        lines.append(" & ".join(cells) + r" \\")

    gm = geomean(speedups)
    geo_cell = f"{gm:.1f}$\\times$" if gm else TBD
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{4}{l}{\textbf{Geometric mean speedup:}} & \textbf{" + geo_cell + r"} \\"
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
    items = filter_items_by_role(CORE_MODELS.items(), args.audience)
    body = render(payload, items=items)
    if args.dry_run:
        print(body)
        return 0
    out = write_generated("nf" + audience_suffix(args.audience), body)
    print(f"[emit] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
