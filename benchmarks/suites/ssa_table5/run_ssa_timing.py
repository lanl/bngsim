#!/usr/bin/env python3
"""Orchestrate the ssa_table5 four-engine exact-SSA timing matrix (14 models x 4 engines).

Each (engine, model) cell runs as an isolated subprocess (_ssa_cell.py) so a native crash
or a stuck C call kills only that cell. 6 concurrent workers, cheap->expensive submission,
per-cell hard wall cap (CELL_WALL_CAP_SEC) as the SIGKILL backstop, incremental save after
every cell. N/A cells (coverage authority) are synthesized directly, not spawned.

  BALLPARK: 6-way concurrency means these wall times are mildly contention-affected. The
  final table needs a clean SERIAL re-run (--workers 1). Flagged in the output.

Outputs (default results/ssa_timing_ballpark.json):
  * per-(model, engine) cell records
  * per-model speedup-vs-bngsim summary (warm median)
  * 14x4 coverage matrix (ran / N/A / too_slow / sim_fail)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import _ssa_config as C

HERE = Path(__file__).resolve().parent
PY = sys.executable  # the venv python running this orchestrator
CELL = str(HERE / "_ssa_cell.py")
JOBOUT = HERE / "results" / "_jobout"


def _hardware() -> dict:
    import platform

    info = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        if platform.system() == "Darwin":
            info["cpu"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            info["physical_cores"] = int(
                subprocess.run(
                    ["sysctl", "-n", "hw.physicalcpu"], capture_output=True, text=True, timeout=5
                ).stdout.strip()
            )
    except Exception:
        pass
    return info


def run_cell(engine: str, model_key: str, warm_override: int | None) -> dict:
    """Spawn one cell subprocess, enforce the hard wall cap, return its record."""
    JOBOUT.mkdir(parents=True, exist_ok=True)
    outf = JOBOUT / f"{model_key}__{engine}.json"
    if outf.exists():
        outf.unlink()
    env = None
    if warm_override is not None:
        import os

        env = {**os.environ, "SSA_WARM_OVERRIDE": str(warm_override)}
    cmd = [PY, CELL, engine, model_key, str(outf)]
    t0 = time.perf_counter()
    killed = False
    stderr_tail = ""
    rc = 0
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=C.CELL_WALL_CAP_SEC, env=env
        )
        rc = proc.returncode
        _lines = (proc.stderr or "").strip().splitlines()
        stderr_tail = _lines[-1][:200] if _lines else ""
    except subprocess.TimeoutExpired:
        killed = True
    wall = time.perf_counter() - t0

    if outf.exists():
        try:
            rec = json.loads(outf.read_text())
        except Exception:
            rec = None
    else:
        rec = None

    if rec is None:
        # Cell wrote nothing: killed on cold (too_slow) or crashed before first flush.
        status = "too_slow" if killed else "sim_fail"
        err = (
            f"hard wall cap {C.CELL_WALL_CAP_SEC:.0f}s exceeded (cold did not complete)"
            if killed
            else f"cell crashed (rc={rc}) {stderr_tail}"
        )
        _, reason = C.cell_status(model_key, engine)
        return {
            "model": model_key,
            "engine": engine,
            "status": status,
            "error": err,
            "cold_sec": None,
            "warm_median_sec": None,
            "n_warm": 0,
            "events_cold": None,
            "events_warm_median": None,
            "events_self": False,
            "conversion": None,
            "faithfulness_flag": "",
            "notes": [],
            "cell_wall_sec": round(wall, 2),
        }

    # Cell flushed at least the cold row. If it's still "partial", the cell was killed
    # (wall cap) or crashed DURING warm — cold is valid, warm was cut.
    if rec.get("status") == "partial":
        rec["status"] = "ok"
        why = (
            "hard wall cap"
            if killed
            else (f"cell died mid-warm (rc={rc})" if rc else "interrupted")
        )
        rec.setdefault("notes", []).append(f"warm cut short: cold valid, warm incomplete ({why})")
    rec["cell_wall_sec"] = round(wall, 2)
    return rec


def synth_na(model_key: str, engine: str) -> dict:
    _, reason = C.cell_status(model_key, engine)
    return {
        "model": model_key,
        "engine": engine,
        "status": "N/A",
        "na_reason": reason,
        "conversion": None,
        "cold_sec": None,
        "warm_median_sec": None,
        "n_warm": 0,
        "events_cold": None,
        "events_warm_median": None,
        "events_self": False,
        "faithfulness_flag": "",
        "notes": [],
    }


def aggregate(records: list[dict]) -> dict:
    by = {(r["model"], r["engine"]): r for r in records}
    # events reference per model: prefer bngsim self-measured warm-median (else cold),
    # else the prior bngsim_activity.json events. Used to fill events/sec for engines
    # that don't self-report a firing count (run_network / RR / COPASI).
    prior = {}
    pa = HERE / "results" / "bngsim_activity.json"
    if pa.exists():
        for e in json.loads(pa.read_text()).get("results", []):
            if e.get("events") is not None:
                prior[e["name"]] = e["events"]
    ev_ref = {}
    for k in C.MODELS:
        b = by.get((k, "bngsim"))
        if b and b.get("events_warm_median") is not None:
            ev_ref[k] = b["events_warm_median"]
        elif b and b.get("events_cold") is not None:
            ev_ref[k] = b["events_cold"]
        elif k in prior:
            ev_ref[k] = prior[k]
    # fill events/sec on every cell
    for r in records:
        warm = r.get("warm_median_sec")
        cold = r.get("cold_sec")
        t = warm if (warm and warm > 0) else (cold if (cold and cold > 0) else None)
        ev = r.get("events_warm_median") if r.get("events_self") else ev_ref.get(r["model"])
        r["events_ref_used"] = None if r.get("events_self") else ev_ref.get(r["model"])
        r["events_per_sec"] = round(ev / t, 1) if (ev and t) else None

    # per-model speedup vs bngsim (warm median; lower time = faster; speedup = bngsim/engine)
    speedup = {}
    for k in C.MODELS:
        b = by.get((k, "bngsim"))
        bt = b.get("warm_median_sec") if b else None
        row = {}
        for eng in C.ENGINES:
            r = by.get((k, eng))
            wt = r.get("warm_median_sec") if r else None
            row[eng] = {
                "warm_median_sec": wt,
                "speedup_vs_bngsim": (round(bt / wt, 3) if (bt and wt) else None),
                "status": r.get("status") if r else "missing",
            }
        speedup[k] = {"bngsim_warm_median_sec": bt, "engines": row}

    # coverage matrix
    matrix = {}
    for k in C.MODELS:
        matrix[k] = {eng: (by.get((k, eng)) or {}).get("status", "missing") for eng in C.ENGINES}
    counts = {}
    for k in matrix:
        for _eng, sval in matrix[k].items():
            counts[sval] = counts.get(sval, 0) + 1
    return {
        "events_ref": ev_ref,
        "speedup": speedup,
        "coverage_matrix": matrix,
        "status_counts": counts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument(
        "--only", default="", help="comma list of model keys (default: all, cheap->expensive)"
    )
    ap.add_argument("--engines", default=",".join(C.ENGINES))
    ap.add_argument("--warm-override", type=int, default=None)
    ap.add_argument("--out", default="results/ssa_timing_ballpark.json")
    args = ap.parse_args()

    models = [m.strip() for m in args.only.split(",") if m.strip()] or C.ordered_models()
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    outpath = HERE / args.out

    # split into spawn (ok/flag) vs synthesized N/A
    jobs, na_records = [], []
    for k in models:
        for eng in engines:
            cov, _ = C.cell_status(k, eng)
            if cov == "na":
                na_records.append(synth_na(k, eng))
            else:
                jobs.append((eng, k))

    print(
        f"ssa_table5 SSA timing | workers={args.workers} | {len(jobs)} cells to run "
        f"+ {len(na_records)} N/A | per-run cap {C.PER_RUN_CAP_SEC:.0f}s | cell cap {C.CELL_WALL_CAP_SEC:.0f}s"
    )
    if args.warm_override is not None:
        print(f"  warm override N={args.warm_override}")
    print(
        "  BALLPARK (6-way concurrency): final numbers need a clean serial re-run (--workers 1)\n"
    )

    records = list(na_records)
    t0 = time.perf_counter()
    done = 0

    def save():
        agg = aggregate(records)
        payload = {
            "suite": "ssa_table5",
            "engine_set": engines,
            "bngsim_version": __import__("bngsim").__version__,
            "hardware": _hardware(),
            "workers": args.workers,
            "per_run_cap_sec": C.PER_RUN_CAP_SEC,
            "cell_wall_cap_sec": C.CELL_WALL_CAP_SEC,
            "warm_budget_sec": C.WARM_BUDGET_SEC,
            "warm_override": args.warm_override,
            "ballpark": True,
            "ballpark_note": "Wall times collected under 6-way concurrency; final results require a clean serial re-run (--workers 1).",
            "elapsed_sec": round(time.perf_counter() - t0, 1),
            "results": sorted(
                records, key=lambda r: (C.MODELS.get(r["model"], {}).get("order", 99), r["engine"])
            ),
            **agg,
        }
        outpath.parent.mkdir(parents=True, exist_ok=True)
        outpath.write_text(json.dumps(payload, indent=2, default=str))

    save()  # write the N/A rows immediately
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_cell, eng, k, args.warm_override): (eng, k) for (eng, k) in jobs}
        for fut in _as_completed_ordered(futs):
            rec = fut.result()
            records.append(rec)
            done += 1
            wm = rec.get("warm_median_sec")
            cs = rec.get("cold_sec")
            eps = rec.get("events_per_sec")
            print(
                f"[{done:2}/{len(jobs)}] {rec['model']:22} {rec['engine']:12} {rec['status']:9} "
                f"cold={_fmt(cs):>10} warm={_fmt(wm):>10} n_warm={rec.get('n_warm', 0):>2} "
                f"{'ev/s=' + str(eps) if eps else ''} {rec.get('error', '')[:60]}"
            )
            save()  # incremental

    save()
    print(f"\nwall: {time.perf_counter() - t0:.1f}s -> {outpath}")
    agg = aggregate(records)
    print("status counts:", agg["status_counts"])


def _fmt(x):
    return f"{x:.4f}s" if isinstance(x, (int, float)) else "-"


def _as_completed_ordered(futs):
    from concurrent.futures import as_completed

    return as_completed(futs)


if __name__ == "__main__":
    main()
