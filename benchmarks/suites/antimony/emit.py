"""
emit.py
-------
Render Supplementary Table S5 (cross-engine Antimony benchmark) as a
LaTeX fragment.

Table S5 is the four-row geomean summary across 117 hand-crafted
Antimony models from the ssys test corpus.  The numbers come from
``showcase/run_ant_exprtk_3engine.py`` -- the Fig-2 runner that also
times RR and AMICI on every ``.ant``.  The antimony suite's own runner
is correctness-only (G1 load / G2 ODE / G3 RR cross-validation), so the
S5 emitter reads from the showcase results JSON instead.

The four S5 rows are:
* BNGsim ExprTk vs.\\ libRoadRunner
* BNGsim codegen vs.\\ libRoadRunner
* BNGsim ExprTk vs.\\ AMICI
* BNGsim codegen vs.\\ AMICI

The current showcase runner times ExprTk only; the two codegen rows
render as ``\\textit{TBD}`` until showcase grows a codegen pass.

The fragment is written to
``bngsim/benchmarks/reports/generated/antimony.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
SUITES_DIR = BENCH_DIR.parent
SHOWCASE_RESULTS = SUITES_DIR / "showcase" / "results"
sys.path.insert(0, str(BENCH_DIR.parents[1]))  # bngsim/benchmarks/

from _emit import (  # noqa: E402
    TBD,
    add_audience_arg,
    audience_suffix,
    filter_by_role,
    fmt_int,
    read_results,
    write_generated,
)


def _latest_showcase_summary() -> dict:
    """Find the most recent showcase results.json and return its summary."""
    if not SHOWCASE_RESULTS.exists():
        return {}
    candidates = sorted(SHOWCASE_RESULTS.glob("ant_exprtk_3engine_*/results.json"))
    if not candidates:
        return {}
    payload = read_results(candidates[-1])
    return payload.get("summary", {}) if isinstance(payload, dict) else {}


def _fmt_geomean(v) -> str:
    if v is None or not isinstance(v, (int, float)) or v <= 0:
        return TBD
    return f"{v:.2f}$\\times$"


def _default_rows(summary: dict) -> list[dict]:
    """The 4 fixed S5 rows.  Each row is a dict so a future paper_role
    tag can be attached without restructuring."""
    rr_n = summary.get("n_rr_consistent_and_timed")
    amici_n = summary.get("n_amici_consistent_and_timed")
    rr_gm = summary.get("rr_over_bng_geomean")
    amici_gm = summary.get("amici_over_bng_geomean")
    return [
        {
            "label": r"BNGsim ExprTk vs.\ libRoadRunner",
            "n": fmt_int(rr_n),
            "gm": _fmt_geomean(rr_gm),
        },
        {"label": r"BNGsim codegen vs.\ libRoadRunner", "n": TBD, "gm": TBD},
        {
            "label": r"BNGsim ExprTk vs.\ AMICI",
            "n": fmt_int(amici_n),
            "gm": _fmt_geomean(amici_gm),
        },
        {"label": r"BNGsim codegen vs.\ AMICI", "n": TBD, "gm": TBD},
    ]


def render(summary: dict, rows=None) -> str:
    """Render the S5 table.  ``rows`` defaults to the 4 fixed cross-engine
    summary rows; pass a filtered subsequence (see
    :func:`_emit.filter_by_role`) to render a per-audience variant."""
    if rows is None:
        rows = _default_rows(summary)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Comparison & \# Models & Geomean speedup \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(" & ".join([r["label"], r["n"], r["gm"]]) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--showcase-summary",
        type=Path,
        default=None,
        help="Specific showcase results.json to read (default: pick the latest).",
    )
    parser.add_argument("--dry-run", action="store_true")
    add_audience_arg(parser)
    args = parser.parse_args()

    if args.showcase_summary is not None:
        payload = read_results(args.showcase_summary)
        summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    else:
        summary = _latest_showcase_summary()

    if not summary:
        print("[emit] no showcase summary found; rendering TBD rows.")

    rows = filter_by_role(_default_rows(summary), args.audience)
    body = render(summary, rows=rows)
    if args.dry_run:
        print(body)
        return 0
    out = write_generated("antimony" + audience_suffix(args.audience), body)
    print(f"[emit] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
