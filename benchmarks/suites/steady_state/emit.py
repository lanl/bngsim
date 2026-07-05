"""
emit.py
-------
Render the KINSOL steady-state dose-response as a Markdown report fragment.

Reads ``results/steady_state_results.json`` and writes
``bngsim/benchmarks/reports/generated/steady_state.md``.

Table S7: Model | Sp | Scan pts | run_net (s) | BNGsim CVODE (s) |
BNGsim KINSOL (s) | Speedup.  Row set is the runner's ``MODELS``
(4 entries).  The runner today only times the BNGsim path -- run_net
and CVODE-baseline columns render TBD until Phase 6 broadens the timing
comparison.
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
    fmt_time,
    read_results,
    write_generated,
)

sys.path.insert(0, str(BENCH_DIR))
from run import MODELS  # noqa: E402

DEFAULT_RESULTS = BENCH_DIR / "results" / "steady_state_results.json"


def _pos(v) -> float | None:
    """Return a positive number, or ``None`` (treats ``0`` / negative as missing)."""
    if isinstance(v, (int, float)) and v > 0:
        return float(v)
    return None


def render(payload: dict, models=None) -> str:
    """Render the S7 table.  ``models`` defaults to ``MODELS``; pass a
    filtered subsequence (see :func:`_emit.filter_by_role`) to render a
    per-audience variant."""
    if models is None:
        models = MODELS
    by_name = {r["name"]: r for r in (payload.get("models", []) if payload else [])}

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lrrcccc}",
        r"\toprule",
        r"Model & Sp & Scan pts & run\_net (s) & BNGsim CVODE (s) "
        r"& BNGsim KINSOL (s) & Speedup \\",
        r"\midrule",
    ]

    for m in models:
        row = by_name.get(m["name"], {})
        sp = row.get("n_species")
        # The runner stamps wall_time_s as a per-strategy totals dict.  Strategy
        # keys: run_network_total, bngsim_cvode_total, bngsim_kinsol_first_total
        # (and variants).  We pick the strict-newton KINSOL total as the
        # canonical KINSOL cell; the strict-fallback variant -- which only
        # records an *additional* cost when the first KINSOL pass fails -- is
        # left out of the headline number.
        wall = row.get("wall_time_s")
        if not isinstance(wall, dict):
            wall = {}
        run_net_t = _pos(wall.get("run_network_total"))
        cvode_t = _pos(wall.get("bngsim_cvode_total"))
        kinsol_t = _pos(wall.get("bngsim_kinsol_first_total")) or _pos(
            wall.get("bngsim_kinsol_two_tier_total")
        )
        # Speedup: run_net / fastest BNGsim path that has data.
        bngsim_best = None
        for cand in (cvode_t, kinsol_t):
            if cand and (bngsim_best is None or cand < bngsim_best):
                bngsim_best = cand
        speedup_cell = TBD
        if run_net_t and bngsim_best:
            speedup_cell = f"{run_net_t / bngsim_best:.2f}$\\times$"
        cells = [
            m["name"].replace("_", r"\_"),
            str(sp) if isinstance(sp, int) else TBD,
            str(m["n_scan_pts"]),
            fmt_time(run_net_t, decimals=3),
            fmt_time(cvode_t, decimals=3),
            fmt_time(kinsol_t, decimals=3),
            speedup_cell,
        ]
        lines.append(" & ".join(cells) + r" \\")

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
    out = write_generated("steady_state" + audience_suffix(args.audience), body)
    print(f"[emit] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
