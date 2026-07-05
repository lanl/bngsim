"""
calibrate.py
------------
Short calibration run for the multi-model fitting benchmark.

Each benchmark problem is run twice -- once per backend -- with a tiny
fixed iteration budget (``CALIB_ITERS``).  The wall-clock time of each
job is recorded to ``results/_calib_runs.json``.

Two things consume that file:

  * ``problems.py`` tiers each problem ``low`` / ``medium`` / ``high`` by
    its measured cost (see the ``effort`` field there); the ``--effort``
    flag of ``run_jobs.py`` / ``consistency_check.py`` then runs a cheap
    subset of the benchmark.
  * a human sizing each problem's production ``max_iterations``.

A calibration job uses the *production* conf verbatim except for
``max_iterations`` (pinned to ``CALIB_ITERS``) and ``output_dir`` (sent
to ``output/_calib/<backend>``), so the relative ordering of the timings
reflects the production run.  Network/model generation and codegen --
the dominant fixed costs for the larger models -- are paid in full.

Usage:  python harness/calibrate.py   [--only SLUG ...] [--backend ...]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problems import BENCH_DIR, PROBLEMS, Problem  # noqa: E402

RESULTS_DIR = BENCH_DIR / "results"
DEFAULT_RESULTS = RESULTS_DIR / "_calib_runs.json"

# Tiny budget: enough that network/model generation, codegen and a few
# scatter-search iterations are all exercised, cheap enough to run the
# non-skipped jobs in one sitting.
CALIB_ITERS = 3

# Problems too expensive to calibrate even at CALIB_ITERS -- their network
# generation alone dominates. kozer_egfr builds a 913-species / ~12k-
# reaction network (~8 min of run_network per job).
# It is unambiguously the highest-effort row and is tiered "high" by fiat,
# not by measurement. calibrate.py skips it unless --include-skipped is
# passed. To recalibrate one anyway: --only <slug> --include-skipped.
CALIB_SKIP = {"kozer_egfr"}

# (conf attribute on Problem, backend tag)
BACKENDS = [("subprocess_conf", "subprocess"), ("bngsim_conf", "bngsim")]


@dataclass
class CalibResult:
    problem: str  # Problem.slug
    backend: str  # "subprocess" | "bngsim"
    wall_clock_s: float
    status: str  # "ok" | "failed"
    notes: str = ""


def _make_calib_conf(prob: Problem, conf_attr: str, backend: str) -> tuple[Path, Path]:
    """Write a ``CALIB_ITERS``-iteration variant of a production conf.

    Returns ``(conf_path, output_dir)``.  Only ``max_iterations`` and
    ``output_dir`` are overridden; everything else is verbatim.
    """
    src: Path = getattr(prob, conf_attr)
    out_rel = f"output/_calib/{backend}"
    kept = []
    for line in src.read_text().splitlines():
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key in ("max_iterations", "output_dir"):
            continue
        kept.append(line)
    kept += [
        "",
        "# --- calibration overrides (calibrate.py) ---",
        f"output_dir={out_rel}",
        f"max_iterations={CALIB_ITERS}",
        "",
    ]
    conf_path = prob.dir / "conf" / f"_calib_{backend}.conf"
    conf_path.write_text("\n".join(kept))
    return conf_path, (prob.dir / out_rel)


def run_one(
    prob: Problem, conf_attr: str, backend: str, pybnf_cmd: str, bngpath: str
) -> CalibResult:
    conf_path: Path = getattr(prob, conf_attr)
    if not conf_path.exists():
        return CalibResult(prob.slug, backend, 0.0, "failed", f"missing conf: {conf_path}")

    calib_conf, _out = _make_calib_conf(prob, conf_attr, backend)
    env = os.environ.copy()
    if bngpath:
        env["BNGPATH"] = bngpath

    t0 = time.perf_counter()
    proc = subprocess.run(
        [pybnf_cmd, "-c", str(calib_conf), "-o"],
        cwd=prob.dir,
        env=env,
        capture_output=True,
        text=True,
    )
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        return CalibResult(
            prob.slug,
            backend,
            wall,
            "failed",
            f"pybnf exit {proc.returncode}: {proc.stderr.strip()[-300:]}",
        )
    return CalibResult(prob.slug, backend, wall, "ok")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None, help="Calibrate only these slugs.")
    parser.add_argument(
        "--backend",
        choices=["subprocess", "bngsim"],
        default=None,
        help="Calibrate only one backend (default: both).",
    )
    parser.add_argument(
        "--include-skipped",
        action="store_true",
        help=f"Also calibrate the normally-skipped problems ({', '.join(sorted(CALIB_SKIP))}) "
        "-- expensive, opt-in only.",
    )
    parser.add_argument("--pybnf-cmd", default=os.environ.get("PYBNF_CMD", "pybnf"))
    parser.add_argument("--bngpath", default=os.environ.get("BNGPATH", ""))
    parser.add_argument("--results-out", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()

    if not args.bngpath:
        print("[calibrate] WARNING: $BNGPATH unset; subprocess jobs may fail.", file=sys.stderr)

    results: list[CalibResult] = []
    for prob in PROBLEMS:
        if args.only and prob.slug not in args.only:
            continue
        if prob.slug in CALIB_SKIP and not args.include_skipped:
            print(f"[calibrate] {prob.slug}: SKIPPED (too expensive; --include-skipped to force)")
            continue
        for conf_attr, backend in BACKENDS:
            if args.backend and backend != args.backend:
                continue
            print(f"[calibrate] {prob.slug} / {backend} ...", flush=True)
            res = run_one(prob, conf_attr, backend, args.pybnf_cmd, args.bngpath)
            print(
                f"[calibrate]   -> {res.status} wall={res.wall_clock_s:.1f}s {res.notes}",
                flush=True,
            )
            results.append(res)
            args.results_out.parent.mkdir(parents=True, exist_ok=True)
            args.results_out.write_text(json.dumps([asdict(r) for r in results], indent=2))

    print(f"[calibrate] wrote {args.results_out}")
    return 0 if all(r.status == "ok" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
