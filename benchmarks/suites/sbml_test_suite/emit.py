"""
emit.py
-------
Render the SBML Test Suite 3-engine pass rates as a Markdown report fragment.

Reads ``results/sbml_test_suite_results.json`` and writes
``bngsim/benchmarks/reports/generated/sbml_test_suite.md``.

Table S8: Engine | Overall | Events.  Four rows: BNGsim,
libRoadRunner, AMICI, COPASI -- every engine measured here through the
same shared-grading path (GH #225), so the figures are apples-to-apples
and no number is quoted from literature.  The third column from the
predecessor table -- "Candidates" (ssys BioModels) -- was dropped per the
locked Phase-5 decision; the 506-cross-validated BioModels result lives in
Fig S1 (``suites/biomodels``) instead.

Cells with no data (engine not run in this sweep) render as
``\\textit{TBD}\\%``.  The events subset counts cases whose tag list
contains a tag starting with ``Event`` (excluding ``CSymbolDelay``) -- the
convention from ``reference_sbml_test_suite`` and the runner's harness.
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
    fmt_int,
    fmt_pct,
    read_results,
    write_generated,
)

DEFAULT_RESULTS = BENCH_DIR / "results" / "sbml_test_suite_results.json"


def _is_event_case(case: dict) -> bool:
    tags = case.get("tags") or []
    return any(t.startswith("Event") and not t.startswith("CSymbolDelay") for t in tags)


def _engine_stats(payload: dict, engine: str) -> tuple[float | None, float | None]:
    """Return (overall_pct, events_pct) for ``engine``.

    Overall comes from summary.<engine>.{pass,tested}; events is filtered
    from the per-case list using the ``status``/``<engine>_status`` columns.
    An engine not run in this sweep (or with no in-scope cases) returns
    ``(None, None)`` and renders as TBD.
    """
    summary = (payload.get("summary") or {}).get(engine, {})
    tested = summary.get("tested")
    passed = summary.get("pass")
    overall_pct = None
    if tested and isinstance(tested, int) and tested > 0:
        overall_pct = 100.0 * (passed or 0) / tested

    # Events subset: re-tally from cases using the engine-specific status field.
    status_key = "status" if engine == "bngsim" else f"{engine}_status"
    e_total = 0
    e_pass = 0
    for c in payload.get("cases", []):
        if not _is_event_case(c):
            continue
        st = c.get(status_key)
        if st in (None, "skipped"):
            continue
        e_total += 1
        if st == "pass":
            e_pass += 1
    events_pct = (100.0 * e_pass / e_total) if e_total else None
    return overall_pct, events_pct


def _engine_present(payload: dict, engine: str) -> bool:
    return engine in (payload.get("engines") or [])


# The fixed engine rows of S8, in display order. Label widths are padded so the
# rendered LaTeX column aligns; the trailing engine key drives stat lookup.
_S8_ENGINES = [
    ("BNGsim       ", "bngsim"),
    ("libRoadRunner", "rr"),
    ("AMICI        ", "amici"),
    ("COPASI       ", "copasi"),
]


def _default_rows(payload: dict) -> list[dict]:
    """The fixed engine rows of S8.  Each row is a dict so a future
    ``paper_role`` tag can be attached without restructuring.  An engine not
    run in this sweep renders TBD (None stats)."""
    rows = []
    for label, key in _S8_ENGINES:
        if _engine_present(payload, key):
            overall, events = _engine_stats(payload, key)
        else:
            overall = events = None
        rows.append({"label": label, "overall": overall, "events": events})
    return rows


def render(payload: dict, rows=None) -> str:
    """Render the S8 table.  ``rows`` defaults to the 3 fixed engine
    rows; pass a filtered subsequence (see :func:`_emit.filter_by_role`)
    to render a per-audience variant."""
    if rows is None:
        rows = _default_rows(payload)

    n_total = payload.get("n_cases") or len(payload.get("cases") or [])
    n_events = sum(1 for c in payload.get("cases", []) if _is_event_case(c))

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Engine & Overall$^a$ & Events$^b$ \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            r["label"] + r" & " + fmt_pct(r["overall"]) + r" & " + fmt_pct(r["events"]) + r" \\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"",
        r"\noindent",
        r"$^a$SBML Test Suite: "
        + (fmt_int(n_total) if n_total else TBD)
        + r" cases tested (excluding non-TimeCourse cases).\\",
        r"$^b$Event-tagged subset: "
        + (fmt_int(n_events) if n_events else TBD)
        + r" cases tested.",
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
    payload = payload if isinstance(payload, dict) else {}
    rows = filter_by_role(_default_rows(payload), args.audience)
    body = render(payload, rows=rows)
    if args.dry_run:
        print(body)
        return 0
    out = write_generated("sbml_test_suite" + audience_suffix(args.audience), body)
    print(f"[emit] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
