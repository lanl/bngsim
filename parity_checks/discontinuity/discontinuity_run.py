#!/usr/bin/env python3
"""The discontinuity suite's nightly sweep (issue #1912, phase 0 of #1781).

Runs every case over its full axes (parameter rows x sensitivity columns x
output grids x tolerances), judges each point against its reference, adds the
convergence rows, and writes a ``_core`` report. ``nightly-parity.yml``'s
``discontinuity`` job hands that report to ``compare.sh``, which diffs it
against ``parity_checks/nightly/baselines/discontinuity.json`` and the previous
night, as every other leg is diffed. The report is the verdict; this script
exits 0 whatever the rows say.

A job is one case's parameter row: its points share one reference solve,
which for a branch-by-branch reference is the costly part. Each job runs in a
fresh process (``_rr_common.schedule``) under a wall-clock cap. A job that
overruns the cap, dies, or meets a fault in the harness reports TIMEOUT or
EXCEPTION on every row it owed, so no row drops out of the report silently.
Each run also passes ``timeout=`` to bngsim (``Case.timeout``), so a stalled
point fails as TIMEOUT while the rest of its row still runs.

    uv run --no-sync python parity_checks/discontinuity/discontinuity_run.py \\
        --workers 4 --out "$RUNNER_TEMP/discontinuity.json"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent, HERE.parent / "rr_parity"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _rr_common  # noqa: E402
from _core import JobResult, tally, versions, write_report  # noqa: E402
from _core.versions import git_rev  # noqa: E402

#: The cap on one job, a case's row: up to 40 points and their convergence
#: rows, and the reference solve. Each point is also cut by bngsim's own
#: ``timeout`` (``Case.timeout``), so a stall costs its point that long and no
#: more (#1779's stalls are cut at 10 s). The 151 jobs take about 64 s on four
#: workers on an M4 Max; the cap is for a job that hangs outside a run, and
#: three of them on one worker fit the workflow's 60 minutes.
DEFAULT_TIMEOUT = 900.0


def job_keys(case, row) -> list[tuple[str, str]]:
    """Every (model_id, method) one row's job reports, in report order."""
    points = case.points(row)
    out = [(case.row_id(p.row), p.method) for p in points]
    if case.judge.converge:
        out += [tuple(case.converge_key(g).split("|", 1)) for g in case.groups(points)]
    return out


def _worker(spec: dict, q) -> None:
    try:
        import _sweep as S
        from _cases import BY_NAME

        case = BY_NAME[spec["case"]]
        row = tuple((k, float(v)) for k, v in spec["row"])
        rows = S.run_row(case, row)
        q.put(
            {
                "key": spec["key"],
                "status": "ok",
                "results": [r.to_result(spec["versions"]) for r in rows],
            }
        )
    except BaseException as e:  # noqa: BLE001 - a fault anywhere is the job's row
        q.put({"key": spec["key"], "status": "error", "error": f"{type(e).__name__}: {e}"})


def _filled(spec: dict, outcome: str, note: str, subclass: str) -> list[dict]:
    return [
        {
            "model_id": m,
            "method": meth,
            "reference_engine": "discontinuity",
            "outcome": outcome,
            "exception": note,
            "subclass": subclass,
            "versions": spec["versions"],
        }
        for m, meth in spec["keys"]
    ]


def to_results(spec: dict, res: dict) -> list[dict]:
    """A job's result as report rows, every row it owed included."""
    status = res.get("status")
    if status == "ok":
        got = res["results"]
        have = {(r["model_id"], r["method"]) for r in got}
        missing = [k for k in spec["keys"] if k not in have]
        if missing:
            got = got + _filled(
                {**spec, "keys": missing},
                "EXCEPTION",
                "the job did not report this row",
                "harness",
            )
        return got
    if status == "timeout":
        return _filled(
            spec, "TIMEOUT", f"the job passed its cap of {spec['timeout']:.0f} s", "job"
        )
    if status == "error":
        return _filled(spec, "EXCEPTION", res.get("error", ""), "harness")
    return _filled(spec, "EXCEPTION", f"the job's process died ({res})", "dead")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", default="", help="comma-separated case names (default: all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="per job (s)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from _cases import BY_NAME, CASES

    names = [n for n in args.cases.split(",") if n]
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        ap.error(f"unknown cases {unknown}; the suite holds {sorted(BY_NAME)}")
    cases = [BY_NAME[n] for n in names] if names else list(CASES)
    stamp = versions.stamp()
    specs = [
        {
            "key": case.row_id(row),
            "case": case.name,
            "row": [list(x) for x in row],
            "keys": job_keys(case, row),
            "versions": stamp,
            "timeout": args.timeout,
        }
        for case in cases
        for row in case.rows()
    ]
    t0 = time.perf_counter()

    def on_done(finished, total, res):
        print(f"  [{finished}/{total}] {res.get('status', '?'):8} {res.get('key')}", flush=True)

    raw = _rr_common.schedule(
        specs, _worker, workers=args.workers, timeout_of=lambda s: s["timeout"], on_done=on_done
    )
    rows = [r for spec, res in zip(specs, raw, strict=True) for r in to_results(spec, res)]
    results = [JobResult.from_dict(r) for r in rows]
    meta = {
        "suite": "discontinuity",
        "reference_engine": "discontinuity",
        "regime": "exact_reference",
        "git_rev": git_rev(str(HERE)),
        "versions": stamp,
        "tally": tally(r.outcome for r in results),
        "n_jobs": len(specs),
        "n_rows": len(results),
        "cases": [c.name for c in cases],
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "hardware": _rr_common.hardware_info(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_report(out, results, meta=meta)
    print(f"report: {out}  {meta['tally']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
