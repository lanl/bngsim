"""
emit.py
-------
Render the forward sensitivity + FIM benchmark as a Markdown report fragment.

Reads ``results/forward_sens_results.json`` and writes
``bngsim/benchmarks/reports/generated/forward_sens.md``.

Table S9 columns: Model | Sp | Params | Serial BNGsim |
{2 cores: CPU, Wall} | {4 cores: CPU, Wall} | {8 cores: CPU, Wall} |
AMICI.  Times are in seconds.

The current runner uses the ``staggered`` corrector method; the
``sharded`` sub-dict is empty when ``--no-sharded`` is set (the default
under ``--mode correctness``), so the per-core columns render as TBD
until a timing-mode run lands.
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
from run import TARGET_MODELS  # noqa: E402

DEFAULT_RESULTS = BENCH_DIR / "results" / "forward_sens_results.json"

# Core counts shown in the per-core blocks of the table.
CORE_COUNTS = (2, 4, 8)


def _method_payload(row: dict) -> dict:
    """Return the per-method payload for the staggered (or first) method."""
    pm = row.get("per_method", {}) if row else {}
    if "staggered" in pm:
        return pm["staggered"]
    return next(iter(pm.values()), {}) if pm else {}


def _ms_to_s(v) -> float | None:
    if not isinstance(v, (int, float)) or v <= 0:
        return None
    return v / 1000.0


def _shard_pair(method: dict, ncores: int) -> tuple[float | None, float | None]:
    """Return (cpu_s, wall_s) for the given core count, or (None, None)."""
    sharded = method.get("sharded", {})
    if not isinstance(sharded, dict):
        return None, None
    entry = sharded.get(str(ncores)) or sharded.get(ncores) or {}
    return _ms_to_s(entry.get("cpu_ms")), _ms_to_s(entry.get("wall_ms"))


def render(payload: dict, names=None) -> str:
    """Render the S9 table.  ``names`` defaults to ``TARGET_MODELS``;
    pass a filtered subsequence to render a per-audience variant.

    Note: today's ``TARGET_MODELS`` is a list of bare strings, so no
    ``paper_role`` is attached -- ``--audience main`` therefore renders
    an empty body, ``--audience supp`` keeps all rows.  Converting
    ``TARGET_MODELS`` to a list of dicts is a separate refinement.
    """
    if names is None:
        names = TARGET_MODELS
    by_name = {r["name"]: r for r in (payload.get("models", []) if payload else [])}

    col_spec = "lrr" + "c" * (1 + 2 * len(CORE_COUNTS) + 1)
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{" + col_spec + r"}",
        r"\toprule",
        r" & & & Serial "
        + "".join(r"& \multicolumn{2}{c}{" + f"{n} cores" + r"} " for n in CORE_COUNTS)
        + r"& \\",
        " ".join(
            r"\cmidrule(lr){" + f"{5 + 2 * i}-{6 + 2 * i}" + r"}" for i in range(len(CORE_COUNTS))
        ),
        r"Model & Sp & Params & BNGsim "
        + "".join(r"& CPU & Wall " for _ in CORE_COUNTS)
        + r"& AMICI \\",
        r"\midrule",
    ]

    for name in names:
        row = by_name.get(name, {})
        method = _method_payload(row)
        sp = row.get("n_species")
        n_params = row.get("n_params") or row.get("n_params_net")
        serial_s = _ms_to_s(method.get("bngsim_serial_ms"))
        amici_s = _ms_to_s(method.get("amici_ms"))

        cells = [
            name.replace("_", r"\_"),
            str(sp) if isinstance(sp, int) else TBD,
            str(n_params) if isinstance(n_params, int) else TBD,
            fmt_time(serial_s, decimals=3),
        ]
        for n in CORE_COUNTS:
            cpu_s, wall_s = _shard_pair(method, n)
            cells.append(fmt_time(cpu_s, decimals=3))
            cells.append(fmt_time(wall_s, decimals=3))
        cells.append(fmt_time(amici_s, decimals=3))
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
    names = filter_by_role(TARGET_MODELS, args.audience)
    body = render(payload if isinstance(payload, dict) else {}, names=names)
    if args.dry_run:
        print(body)
        return 0
    out = write_generated("forward_sens" + audience_suffix(args.audience), body)
    print(f"[emit] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
