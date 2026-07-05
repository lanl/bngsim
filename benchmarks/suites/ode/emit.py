"""
emit.py
-------
Render the ODE benchmark as a Markdown report fragment.

Reads ``results/ode_results.json`` and writes
``bngsim/benchmarks/reports/generated/ode.md``.

Supplementary Table S1 has six columns: Model, Sp, Rxns, BNGsim,
run_net, RR, AMICI.  The ``ode`` suite runner today only measures BNGsim
vs. run_network; the RR / AMICI columns render as ``\\textit{TBD}``
until Phase 6 broadens the suite to four engines.  Cells with no timing
yet (``mode=correctness`` runs) also render as TBD.

The row set comes from the runner's ``CURATED_MODELS`` registry so the
table layout stays stable across runs.
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

# Importing the runner's curated registry costs no heavy deps -- ode/run.py
# defers ``import bngsim`` to inside its timing functions.
sys.path.insert(0, str(BENCH_DIR))
from run import CURATED_MODELS  # noqa: E402

DEFAULT_RESULTS = BENCH_DIR / "results" / "ode_results.json"


def _by_name(payload) -> dict:
    """Index a results JSON's ``results`` list by model name."""
    rows = payload.get("results", []) if isinstance(payload, dict) else []
    return {r["name"]: r for r in rows if isinstance(r, dict) and "name" in r}


def _bn_timing(row) -> float | None:
    """BNGsim median wall-clock for a result row, if present."""
    t = row.get("timing", {}) if row else {}
    return t.get("bngsim_median")


def _rn_timing(row) -> float | None:
    t = row.get("timing", {}) if row else {}
    return t.get("run_network_median")


def render(payload: dict, models=None) -> str:
    """Render the S1 table.  ``models`` defaults to ``CURATED_MODELS``;
    pass a filtered subsequence (see :func:`_emit.filter_by_role`) to
    render a per-audience variant."""
    if models is None:
        models = CURATED_MODELS
    by_name = _by_name(payload)

    lines = [
        r"\begin{small}",
        r"\begin{longtable}{lrrcccc}",
        r"\toprule",
        r"Model & Sp & Rxns & BNGsim (s) & run\_net (s) & RR (s) & AMICI (s) \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Model & Sp & Rxns & BNGsim (s) & run\_net (s) & RR (s) & AMICI (s) \\",
        r"\midrule",
        r"\endhead",
    ]

    bn_for_geomean: list[float] = []
    rn_for_geomean: list[float] = []
    for m in models:
        row = by_name.get(m["name"], {})
        bn_t = _bn_timing(row)
        rn_t = _rn_timing(row)
        if bn_t and rn_t and bn_t > 0 and rn_t > 0:
            bn_for_geomean.append(bn_t)
            rn_for_geomean.append(rn_t)

        # Model name: TeX-escape underscores.
        name_tex = m["name"].replace("_", r"\_")
        sp = row.get("species") if row else None
        rxns = row.get("reactions") if row else None
        cells = [
            name_tex,
            str(sp) if sp is not None else TBD,
            str(rxns) if rxns is not None else TBD,
            fmt_time(bn_t),
            fmt_time(rn_t),
            TBD,  # RR -- Phase 6
            TBD,  # AMICI -- Phase 6
        ]
        lines.append(" & ".join(cells) + r" \\")

    # Geomean speedup row: run_network / BNGsim for the rows that have both.
    bn_gm = geomean(bn_for_geomean)
    rn_gm = geomean(rn_for_geomean)
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{4}{l}{\textbf{Geometric mean speedup vs.\ BNGsim:}} "
        + r"& \textbf{"
        + fmt_speedup(rn_gm, bn_gm)
        + r"} & \textbf{"
        + TBD
        + r"} & \textbf{"
        + TBD
        + r"} \\"
    )
    lines += [
        r"\bottomrule",
        r"\end{longtable}",
        r"\end{small}",
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
    models = filter_by_role(CURATED_MODELS, args.audience)
    body = render(payload if isinstance(payload, dict) else {}, models=models)
    if args.dry_run:
        print(body)
        return 0
    out = write_generated("ode" + audience_suffix(args.audience), body)
    print(f"[emit] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
