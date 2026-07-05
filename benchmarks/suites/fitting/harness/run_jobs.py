"""
run_jobs.py
-----------
Drive the multi-model fitting benchmark for the bngsim paper.

For every problem in ``problems.PROBLEMS`` this runs two PyBioNetFit
jobs -- ``conf/subprocess.conf`` (legacy BNG2.pl / run_network / NFsim)
and ``conf/bngsim.conf`` (in-process BNGsim) -- captures wall-clock time
and the final objective, and writes a results JSON that
``../emit.py`` turns into one table row per problem.

The two confs of a problem differ only in ``bngl_backend`` and
``output_dir``; they run the same scatter search with the same fixed
iteration budget, so the benchmark measures wall-clock for identical
fitting work.

PyBNF writes the best-fit objective to
``<output_dir>/Results/sorted_params_final.txt``: a tab-separated table
sorted by ascending objective, so the first data row is the best fit.
``parse_final_objective`` returns that value. (Verified against a
completed PyBNF v1.3.0 run; ``sorted_params_<n>.txt`` are per-iteration
snapshots and ``sorted_params_backup.txt`` the crash-recovery copy.)

Wall-clock covers the whole PyBNF process: model/network generation,
codegen, and the scatter-search loop. Results stream to disk after every
job, so an interrupted run is resumable via ``--only``.

PyBNF finds BNG2.pl via ``$BNGPATH``; this harness forwards it to each
subprocess.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problems import (  # noqa: E402
    BENCH_DIR,
    EFFORT_LEVELS,
    PROBLEMS,
    Problem,
    problems_at_effort,
)

RESULTS_DIR = BENCH_DIR / "results"
DEFAULT_RESULTS = RESULTS_DIR / "fitting_runs.json"

# (conf attribute on Problem, backend tag, human label)
BACKENDS = [
    ("subprocess_conf", "subprocess", "BNG2.pl / run_network subprocess"),
    ("bngsim_conf", "bngsim", "in-process BNGsim"),
]


@dataclass
class JobResult:
    problem: str  # Problem.slug
    label: str  # Problem.label
    family: str  # "ode" | "ssa" | "nf"
    n_params: int
    backend: str  # "subprocess" | "bngsim"
    wall_clock_s: float
    final_objective: float
    status: str  # "ok" | "failed" | "skipped"
    notes: str = ""


def _resolve_output_dir(conf_path: Path, problem_dir: Path) -> Path:
    """Extract ``output_dir=`` from a conf, resolved against the problem dir."""
    pattern = re.compile(r"^\s*output_dir\s*=\s*(.+?)\s*$")
    for line in conf_path.read_text().splitlines():
        m = pattern.match(line)
        if m:
            p = Path(m.group(1).split("#")[0].strip())
            return p if p.is_absolute() else (problem_dir / p).resolve()
    raise ValueError(f"No output_dir= directive in {conf_path}")


_RESULT_FILES = ("sorted_params_final.txt", "sorted_params_backup.txt")


def parse_final_objective(output_dir: Path) -> float:
    """Best (smallest) objective from a PyBNF run dir.

    Reads ``Results/sorted_params_final.txt`` (fallback
    ``sorted_params_backup.txt``); the first data row holds the best fit.
    The 'Obj' column index is read from the header.
    """
    results_dir = output_dir / "Results"
    sorted_file = next(
        (results_dir / c for c in _RESULT_FILES if (results_dir / c).exists()), None
    )
    if sorted_file is None:
        raise ValueError(f"No result file ({', '.join(_RESULT_FILES)}) in {results_dir}")
    lines = [ln.rstrip("\n") for ln in sorted_file.read_text().splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"{sorted_file} has no data rows")
    header = [c.strip() for c in lines[0].lstrip("#").split("\t") if c.strip()]
    try:
        obj_idx = header.index("Obj")
    except ValueError as err:
        raise ValueError(f"No 'Obj' column in {sorted_file} header: {header!r}") from err
    data = [c.strip() for c in lines[1].split("\t") if c.strip() != ""]
    if obj_idx >= len(data):
        raise ValueError(f"Row in {sorted_file} has {len(data)} cols, expected {obj_idx}")
    return float(data[obj_idx])


def run_one(
    prob: Problem, conf_attr: str, backend: str, pybnf_cmd: str, bngpath: str
) -> JobResult:
    """Run one PyBioNetFit job and capture timing + final objective."""
    base = JobResult(
        prob.slug, prob.label, prob.family, prob.n_params, backend, 0.0, float("nan"), "skipped"
    )
    conf_path: Path = getattr(prob, conf_attr)
    if not conf_path.exists():
        base.notes = f"missing conf: {conf_path}"
        return base

    output_dir = _resolve_output_dir(conf_path, prob.dir)
    env = os.environ.copy()
    if bngpath:
        env["BNGPATH"] = bngpath

    t0 = time.perf_counter()
    proc = subprocess.run(
        [pybnf_cmd, "-c", str(conf_path), "-o"],
        cwd=prob.dir,
        env=env,
        capture_output=True,
        text=True,
    )
    base.wall_clock_s = time.perf_counter() - t0

    if proc.returncode != 0:
        base.status = "failed"
        base.notes = f"pybnf exit {proc.returncode}: {proc.stderr.strip()[-400:]}"
        return base
    try:
        base.final_objective = parse_final_objective(output_dir)
    except (ValueError, OSError) as err:
        base.status = "failed"
        base.notes = f"parse: {err}"
        return base
    base.status = "ok"
    return base


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None, help="Run only these problems (slugs).")
    parser.add_argument(
        "--backend",
        choices=["subprocess", "bngsim"],
        default=None,
        help="Run only one backend (default: both).",
    )
    parser.add_argument(
        "--effort",
        choices=list(EFFORT_LEVELS),
        default="high",
        help="Cumulative effort threshold: 'low' runs only low-effort "
        "problems, 'medium' runs low + medium, 'high' runs all "
        "(default: high -- the full sweep).",
    )
    parser.add_argument(
        "--pybnf-cmd",
        default=os.environ.get("PYBNF_CMD", "pybnf"),
        help="pybnf CLI path (default: $PYBNF_CMD or 'pybnf').",
    )
    parser.add_argument(
        "--bngpath",
        default=os.environ.get("BNGPATH", ""),
        help="BioNetGen distribution dir (default: $BNGPATH).",
    )
    parser.add_argument(
        "--results-out",
        type=Path,
        default=DEFAULT_RESULTS,
        help="Where to write captured timings/objectives.",
    )
    args = parser.parse_args()

    if not args.bngpath:
        print("[run_jobs] WARNING: $BNGPATH unset; subprocess jobs may fail.", file=sys.stderr)

    # Resume-friendly: keep any prior results for jobs we are not re-running.
    prior: dict[tuple[str, str], dict] = {}
    if args.results_out.exists():
        for r in json.loads(args.results_out.read_text()):
            prior[(r["problem"], r["backend"])] = r

    # Cumulative effort threshold: 'low' < 'medium' < 'high' (= all).
    selected = {p.slug for p in problems_at_effort(args.effort)}
    print(
        f"[run_jobs] effort={args.effort}: {len(selected)} of {len(PROBLEMS)} problems",
        flush=True,
    )

    results: list[JobResult] = []
    ran: list[JobResult] = []
    for prob in PROBLEMS:
        run_this = prob.slug in selected and not (args.only and prob.slug not in args.only)
        if not run_this:
            # Preserve prior records for problems we are not re-running.
            for _, backend, _ in BACKENDS:
                if (prob.slug, backend) in prior:
                    results.append(JobResult(**prior[(prob.slug, backend)]))
            continue
        for conf_attr, backend, label in BACKENDS:
            if args.backend and backend != args.backend:
                if (prob.slug, backend) in prior:
                    results.append(JobResult(**prior[(prob.slug, backend)]))
                continue
            print(f"[run_jobs] {prob.slug} / {backend} -- {label}", flush=True)
            res = run_one(prob, conf_attr, backend, args.pybnf_cmd, args.bngpath)
            results.append(res)
            ran.append(res)
            print(
                f"[run_jobs]   -> {res.status} wall={res.wall_clock_s:.1f}s "
                f"obj={res.final_objective}",
                flush=True,
            )
            args.results_out.parent.mkdir(parents=True, exist_ok=True)
            args.results_out.write_text(json.dumps([asdict(r) for r in results], indent=2))

    print(f"[run_jobs] wrote {args.results_out}")
    return 0 if all(r.status == "ok" for r in ran) else 1


if __name__ == "__main__":
    raise SystemExit(main())
