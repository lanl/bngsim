"""
emit.py
-------
Render Supplementary Table S6 (pythonic workflow comparison) as a LaTeX
fragment.

S6 is the joint ODE+SSA pythonic-workflow comparison.  This emitter is
hosted under the ``python_ode`` suite (the ODE half is the meatier
table; ``python_ssa`` intentionally has no ``emit.py``) and reads both
``../python_ode/results/python_ode_results.json`` and
``../python_ssa/results/python_ssa_results.json``.

Shape: one row per (engine, input) pair, one column per model.  The
runners' current model set determines the columns:
* ODE side -- whatever models ``python_ode`` ran (ModelBuilder-only
  after the Phase-4 rebuild: ``simple_system``, ``m1_ground``, ``sir``,
  ``lotka_volterra`` typically);
* SSA side -- ``python_ssa`` runs ``simple_system`` only (its ``species``
  is documented in the runner README), so non-``simple_system`` cells in
  the gillespy2 rows render as ``n/a``.

Cells with no timing yet render as ``\\textit{TBD}``.  The fragment is
written to ``bngsim/benchmarks/reports/generated/python_ode.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
SUITES_DIR = BENCH_DIR.parent
sys.path.insert(0, str(BENCH_DIR.parents[1]))  # bngsim/benchmarks/

from _emit import (  # noqa: E402
    add_audience_arg,
    audience_suffix,
    filter_by_role,
    fmt_time,
    read_results,
    write_generated,
)

DEFAULT_ODE = SUITES_DIR / "python_ode" / "results" / "python_ode_results.json"
DEFAULT_SSA = SUITES_DIR / "python_ssa" / "results" / "python_ssa_results.json"

NA = r"n/a"


def _ode_columns(ode_payload: dict) -> list[str]:
    """Column order = model order in python_ode results."""
    return [m["name"] for m in (ode_payload.get("models", []) if ode_payload else [])]


def _ode_time(ode_payload: dict, model_name: str, engine: str) -> float | None:
    for m in ode_payload.get("models", []) if ode_payload else []:
        if m["name"] != model_name:
            continue
        e = m.get("engines", {}).get(engine)
        if not e:
            return None
        t = e.get("time_s")
        return t if t and t > 0 and e.get("correctness_ok") else None
    return None


def _ssa_time(ssa_payload: dict, model_name: str, engine: str, path: str) -> float | str | None:
    """Return seconds, or ``NA`` if ``model_name`` differs from python_ssa's model."""
    if not ssa_payload:
        return None
    if model_name != ssa_payload.get("model"):
        return NA
    for p in ssa_payload.get("paths", []):
        if p.get("path") != path:
            continue
        key = "bngsim_time_s" if engine == "bngsim" else "gillespy2_time_s"
        t = p.get(key)
        return t if t and t > 0 and p.get("correctness_ok") else None
    return None


def _default_rows(ode_payload: dict, ssa_payload: dict, columns: list[str]) -> list[dict]:
    """Default 5 rows for the joint S6 table.  Each row is a dict so a
    future ``paper_role`` tag can be attached without restructuring."""
    rows: list[dict] = []
    # ODE engines x ModelBuilder (python_ode is ModelBuilder-only after Phase 4).
    for engine_label, engine_key in (
        ("BNGsim", "bngsim"),
        ("scipy/LSODA", "scipy"),
        ("Diffrax", "diffrax"),
    ):
        cells = [fmt_time(_ode_time(ode_payload, c, engine_key), decimals=4) for c in columns]
        rows.append({"engine": engine_label, "input": "ModelBuilder", "cells": cells})

    # SSA engines x {ModelBuilder, NET reader} (python_ssa).
    for engine_label, engine_key in (
        ("BNGsim Gillespie", "bngsim"),
        ("gillespy2 (SSA)", "gillespy2"),
    ):
        for path_label, path_key in (
            ("ModelBuilder", "modelbuilder"),
            ("NET reader", "net_reader"),
        ):
            cells: list[str] = []
            for c in columns:
                t = _ssa_time(ssa_payload, c, engine_key, path_key)
                cells.append(NA if t == NA else fmt_time(t, decimals=4))
            rows.append({"engine": engine_label, "input": path_label, "cells": cells})
    return rows


def render(ode_payload: dict, ssa_payload: dict, rows=None) -> str:
    """Render the S6 joint table.  ``rows`` defaults to the 5 (engine,
    input) rows; pass a filtered subsequence (see
    :func:`_emit.filter_by_role`) to render a per-audience variant."""
    columns = _ode_columns(ode_payload) or ["simple_system"]
    n_cols = len(columns)
    if rows is None:
        rows = _default_rows(ode_payload, ssa_payload, columns)

    header_models = " & ".join(c.replace("_", r"\_") for c in columns)
    col_spec = "ll" + "c" * n_cols
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        r"Engine & Input & " + header_models + r" \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(" & ".join([r["engine"], r["input"], *r["cells"]]) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ode-results", type=Path, default=DEFAULT_ODE)
    parser.add_argument("--ssa-results", type=Path, default=DEFAULT_SSA)
    parser.add_argument("--dry-run", action="store_true")
    add_audience_arg(parser)
    args = parser.parse_args()

    ode_payload = read_results(args.ode_results)
    ssa_payload = read_results(args.ssa_results)
    if not ode_payload:
        print(f"[emit] no ODE results at {args.ode_results}; rendering TBD/NA.")
    if not ssa_payload:
        print(f"[emit] no SSA results at {args.ssa_results}; rendering TBD/NA.")

    ode_payload = ode_payload if isinstance(ode_payload, dict) else {}
    ssa_payload = ssa_payload if isinstance(ssa_payload, dict) else {}
    columns = _ode_columns(ode_payload) or ["simple_system"]
    rows = filter_by_role(_default_rows(ode_payload, ssa_payload, columns), args.audience)
    body = render(ode_payload, ssa_payload, rows=rows)
    if args.dry_run:
        print(body)
        return 0
    out = write_generated("python_ode" + audience_suffix(args.audience), body)
    print(f"[emit] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
