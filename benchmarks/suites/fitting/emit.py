"""
emit.py
-------
Render the multi-model fitting benchmark as a LaTeX table fragment.

Reads ``results/fitting_runs.json`` (or any ``--results-in`` JSON in that
shape -- ``_calib_runs.json`` works too, for a calibration-only render
that fills the wall-clock columns and leaves the objective columns as
``\\textit{TBD}``).

The fragment is written to ``bngsim/benchmarks/reports/generated/fitting.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BENCH_DIR / "harness"))
sys.path.insert(0, str(BENCH_DIR.parents[1]))  # bngsim/benchmarks/

from _emit import (  # noqa: E402
    TBD,
    add_audience_arg,
    audience_suffix,
    filter_by_role,
    fmt_obj,
    fmt_speedup,
    fmt_time,
    read_results,
    write_generated,
)
from problems import PROBLEMS  # noqa: E402

DEFAULT_RESULTS = BENCH_DIR / "results" / "fitting_runs.json"

FAMILY_NAME = {
    "ode": "BioNetGen ODE",
    "ssa": "BioNetGen SSA",
    "nf": "Network-free (NFsim app vs.\\ RuleMonkey)",
}


def render(by_key: dict, problems=None) -> str:
    """``by_key`` maps ``(problem_slug, backend)`` -> result record.

    ``problems`` defaults to ``PROBLEMS``; pass a filtered subsequence
    (see :func:`_emit.filter_by_role`) to render a per-audience variant.
    """
    if problems is None:
        problems = PROBLEMS
    lines = [
        r"\begin{table}[!t]",
        r"\centering",
        r"\caption{PyBNF scatter-search fitting benchmark across "
        r"BioNetGen example models (PyBNF iScience 2019 Data~S1). Each "
        r"model is fit twice with identical scatter-search settings and a "
        r"fixed iteration budget: once through the legacy BNG2.pl / "
        r"run\_network / NFsim subprocess stack, once through the "
        r"in-process BNGsim backend. Wall-clock covers the whole PyBNF "
        r"run (model and network generation, codegen, and the fit loop). "
        r"Speedup is subprocess time divided by in-process time.}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{Wall-clock time (s)} "
        r"& \multicolumn{2}{c}{Objective} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}",
        r"Problem & Params & Subprocess & BNGsim & Speedup "
        r"& Subprocess & BNGsim \\",
    ]

    prev_family: str | None = None
    for prob in problems:
        if prob.family != prev_family:
            fam = FAMILY_NAME.get(prob.family, prob.family)
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{7}{l}{\textit{" + fam + r"}} \\")
            prev_family = prob.family
        sub = by_key.get((prob.slug, "subprocess"), {})
        bn = by_key.get((prob.slug, "bngsim"), {})
        sub_ok = sub.get("status") == "ok"
        bn_ok = bn.get("status") == "ok"
        sub_t = sub.get("wall_clock_s") if sub_ok else None
        bn_t = bn.get("wall_clock_s") if bn_ok else None
        row = " & ".join(
            [
                prob.label,
                str(prob.n_params),
                fmt_time(sub_t, decimals=1) if sub_t is not None else TBD,
                fmt_time(bn_t, decimals=1) if bn_t is not None else TBD,
                fmt_speedup(sub_t, bn_t),
                fmt_obj(sub.get("final_objective") if sub_ok else None),
                fmt_obj(bn.get("final_objective") if bn_ok else None),
            ]
        )
        lines.append(row + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:fitting-benchmark}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-in", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--dry-run", action="store_true", help="Print the fragment; do not write.")
    add_audience_arg(parser)
    args = parser.parse_args()

    payload = read_results(args.results_in)
    if not payload:
        print(f"[emit] no results at {args.results_in}; rendering TBD rows.")
        by_key: dict = {}
    else:
        by_key = {(r["problem"], r["backend"]): r for r in payload}

    problems = filter_by_role(PROBLEMS, args.audience)
    body = render(by_key, problems=problems)
    if args.dry_run:
        print(body)
        return 0
    out = write_generated("fitting" + audience_suffix(args.audience), body)
    print(f"[emit] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
