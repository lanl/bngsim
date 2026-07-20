#!/usr/bin/env python3
"""Measure per-model stochastic ACTIVITY (Gillespie events per unit simulated
time) for the ssa_table5 corpus, using BNGsim's exact-SSA engine only.

- One replicate per model (activity is an intensive rate; ensemble not needed).
- Per-run wall cap (default 60 s) via bngsim's own run(timeout=...).
- On timeout, fall back to progressively shorter horizons to still recover the
  activity rate (flagged partial).
- ~N workers in parallel (default 10), process-parallel.

Writes results/bngsim_activity.json and prints a table. BNGsim only for now
(RoadRunner / COPASI columns are a later step).
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = json.loads((HERE / "corpus.json").read_text())
# numeric horizon fallback for SBML rows whose corpus t_end is a "confirm from
# paper" placeholder string -- use the screen (ssa_jobs) horizon.
SBML_PLACEHOLDER_TEND = {"BIOMD0000000478": 10.0, "BIOMD0000000586": 10.0, "BIOMD0000000587": 10.0}


def _load(kind, path):
    import bngsim

    return bngsim.Model.from_net(path) if kind == "bngl" else bngsim.Model.from_sbml(path)


def run_one(job):
    """job = (name, kind, path, t_end, n_steps, cap_sec). Returns a result dict."""
    import bngsim

    name, kind, path, t_end, n_steps, cap = job
    n_points = min(max(int(n_steps or 100), 2), 1001)
    try:
        t0 = time.perf_counter()
        model = _load(kind, str(HERE / path))
        load_sec = time.perf_counter() - t0
        sim = bngsim.Simulator(model, method="ssa")
    except Exception as e:  # noqa: BLE001
        return {
            "name": name,
            "kind": kind,
            "status": "load_error",
            "error": f"{type(e).__name__}: {e}"[:200],
        }

    def _attempt(te):
        model.reset()
        tr = time.perf_counter()
        r = sim.run(t_span=(0.0, float(te)), n_points=n_points, seed=1, timeout=cap)
        wall = time.perf_counter() - tr
        ns = int((r.solver_stats or {}).get("n_steps", 0) or 0)
        backend = (getattr(r, "ssa_diagnostics", None) or {}).get(
            "propensity_backend", "interpreted"
        )
        return ns, wall, backend

    # full horizon first; on timeout, shrink to recover the rate
    horizons = [t_end] + [t_end / (10**k) for k in range(1, 8)]
    for i, te in enumerate(horizons):
        try:
            ns, wall, backend = _attempt(te)
            span = max(float(te), 1e-15)
            return {
                "name": name,
                "kind": kind,
                "status": "ok" if i == 0 else "partial",
                "t_end": te,
                "full_t_end": t_end,
                "events": ns,
                "events_per_time": ns / span,
                "wall_sec": round(wall, 4),
                "load_sec": round(load_sec, 4),
                "backend": backend,
            }
        except bngsim.SimulationTimeout:
            continue
        except Exception as e:  # noqa: BLE001
            return {
                "name": name,
                "kind": kind,
                "status": "error",
                "error": f"{type(e).__name__}: {e}"[:200],
            }
    return {
        "name": name,
        "kind": kind,
        "status": "timeout_all",
        "full_t_end": t_end,
        "note": f"did not complete even at t_end={horizons[-1]:.3g} within {cap}s",
    }


def build_jobs(cap):
    jobs = []
    for m in CORPUS["bngl"]:
        jobs.append((m["name"], "bngl", m["file"], float(m["t_end"]), m.get("n_steps", 100), cap))
    for m in CORPUS["sbml"]:
        te = m["t_end"]
        if not isinstance(te, (int, float)):
            te = SBML_PLACEHOLDER_TEND.get(m["id"], 10.0)
        jobs.append((m["id"], "sbml", m["file"], float(te), m.get("n_points", 100), cap))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=60.0, help="per-run wall cap (s)")
    args = ap.parse_args()

    jobs = build_jobs(args.timeout)
    print(
        f"bngsim {__import__('bngsim').__version__ if False else ''}  "
        f"running {len(jobs)} models  |  workers={args.workers}  cap={args.timeout}s\n"
    )
    results = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, j): j[0] for j in jobs}
        for fut in as_completed(futs):
            results.append(fut.result())
    elapsed = time.perf_counter() - t0

    order = {"ok": 0, "partial": 1, "timeout_all": 2, "error": 3, "load_error": 4}
    results.sort(key=lambda r: (order.get(r["status"], 9), -(r.get("events_per_time") or -1)))
    outdir = HERE / "results"
    outdir.mkdir(exist_ok=True)
    (outdir / "bngsim_activity.json").write_text(
        json.dumps(
            {
                "engine": "bngsim",
                "version": __import__("bngsim").__version__,
                "cap_sec": args.timeout,
                "workers": args.workers,
                "elapsed_sec": round(elapsed, 2),
                "results": results,
            },
            indent=2,
        )
    )

    print(
        f"{'model':24} {'kind':5} {'status':8} {'events/time':>14} {'events':>12} {'t_end':>10} {'wall_s':>8}"
    )
    for r in results:
        ept = r.get("events_per_time")
        ept_s = f"{ept:,.3f}" if ept is not None else "-"
        print(
            f"{r['name']:24} {r['kind']:5} {r['status']:8} {ept_s:>14} "
            f"{str(r.get('events', '-')):>12} {str(r.get('t_end', '-')):>10} {str(r.get('wall_sec', '-')):>8}"
            + (
                f"   {r.get('error') or r.get('note', '')}"
                if r["status"] not in ("ok", "partial")
                else ""
            )
        )
    print(f"\nwall: {elapsed:.1f}s  ->  results/bngsim_activity.json")


if __name__ == "__main__":
    main()
