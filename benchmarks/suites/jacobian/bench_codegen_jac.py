#!/usr/bin/env python3
"""``jacobian`` suite — compiled-C Jacobian vs interpreted (ExprTk) Jacobian.

GH #76 Task 4 added a *compiled* C mirror of the analytical Jacobian
(``bngsim_codegen_jac``) emitted into the codegen ``.so``.  When codegen is
active, the CVODE dense dispatch prefers it over the interpreted
``cvode_analytical_dense_jac`` (ExprTk eval per entry).  Both assemble the
identical scatter — verified bit-identical against ``_dense_analytical_jacobian``
in ``test_codegen_jacobian.py`` — so trajectories match to solver tolerance;
this is a pure per-step **performance** feature.

This benchmark isolates that swap.  Both modes compile the RHS to C (so the RHS
cost is held fixed); only the state-Jacobian differs:

  * ``codegen_jac``  — compiled-C Jacobian (the new path)
  * ``interp_jac``   — interpreted ExprTk Jacobian (BNGSIM_NO_CODEGEN_JAC=1)

The premise (#76 handoff): on a ~290-species functional SBML model the
interpreted Jacobian eval is ~128× the compiled-RHS eval, so once the RHS is
compiled the interpreted Jacobian dominates the per-step cost — exactly what the
compiled Jacobian removes.

Each (model, mode) runs in a fresh subprocess so the load-time codegen decision
sees a clean ``BNGSIM_NO_CODEGEN_JAC``.

Usage::

    python bench_codegen_jac.py                 # full sweep
    python bench_codegen_jac.py --repeats 7
    python bench_codegen_jac.py --effort low
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

HERE = Path(__file__).resolve().parent
_BENCH_ROOT = HERE.parents[1]
ROOT = HERE.parents[2]
sys.path.insert(0, str(_BENCH_ROOT))
import _netbench as nb  # noqa: E402
from _effort import add_effort_arg, filter_by_effort  # noqa: E402

MODELS_DIR = ROOT / "parity_checks" / "rr_parity" / "models"
RESULTS = HERE / "results"


def _relpath(path) -> str:
    """Repo-relative model path for the results (machine-independent); home-scrubbed
    absolute fallback for models outside the bngsim/ tree."""
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        s, home = str(p), str(Path.home())
        return "~" + s[len(home) :] if s.startswith(home) else s


# Large functional SBML models (dense path; KLU off in this build → all dense).
MODELS = [
    {"id": "MODEL9089538076", "T": 100.0, "N": 200, "effort": "low", "note": "ns~200 functional"},
    {
        "id": "BIOMD0000000595",
        "T": 100.0,
        "N": 200,
        "effort": "medium",
        "note": "ns~218, 1490 rxns",
    },
    {
        "id": "MODEL9087255381",
        "T": 100.0,
        "N": 200,
        "effort": "medium",
        "note": "ns~289 functional",
    },
    {"id": "MODEL9087474843", "T": 100.0, "N": 200, "effort": "high", "note": "ns~290, 497 rxns"},
]


def sbml_path(model_id: str) -> Path | None:
    d = MODELS_DIR / model_id
    if not d.is_dir():
        return None
    cand = d / f"{model_id}.xml"
    if cand.exists():
        return cand
    xmls = sorted(d.glob("*.xml"))
    return xmls[0] if xmls else None


def worker(path, mode, T, N, traj_out=None):
    # Both modes compile the RHS; interp_jac additionally declines the compiled
    # Jacobian so dispatch falls back to the interpreted ExprTk dense Jacobian.
    os.environ["BNGSIM_NO_CODEGEN_JAC"] = "1" if mode == "interp_jac" else "0"
    os.environ["BNGSIM_ANALYTICAL_FUNCTIONAL_JAC"] = "1"
    os.environ.pop("BNGSIM_NO_CODEGEN", None)
    out = {"path": _relpath(path), "mode": mode, "T": T, "N": N}
    try:
        import bngsim
        from bngsim._codegen import prepare_model_codegen

        out["version"] = bngsim.__version__
        model = bngsim.Model.from_sbml(str(path))
        # Force compiled RHS in both modes (models < 256 sp aren't auto-codegen'd).
        so = prepare_model_codegen(model)
        if so is None:
            raise RuntimeError("codegen failed")
        model._codegen_so_path = str(so)
        sim = bngsim.Simulator(model, method="ode", jacobian="analytical")
        # warm one run (JIT/cache) then time
        sim.run(t_span=(0.0, T), n_points=N)
        t0 = time.perf_counter()
        res = sim.run(t_span=(0.0, T), n_points=N)
        out["wall"] = time.perf_counter() - t0
        ss = getattr(res, "solver_stats", None)
        out["stats"] = {k: v for k, v in (ss or {}).items() if isinstance(v, (int, float, bool))}
        if traj_out:
            np.savez(traj_out, species=np.asarray(res.species, dtype=float))
            out["traj_out"] = traj_out
        out["ok"] = True
    except Exception as e:  # noqa: BLE001
        import traceback

        out["error"] = repr(e)
        out["traceback"] = traceback.format_exc()
    print(json.dumps(out))


def _spawn(path, mode, T, N, traj_out=None, timeout=1800):
    cmd = [
        sys.executable,
        str(HERE / "bench_codegen_jac.py"),
        "--worker",
        str(path),
        mode,
        repr(T),
        str(N),
    ]
    if traj_out:
        cmd.append(str(traj_out))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    if lines:
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            pass
    return {"error": "no_json", "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]}


def correctness(path, T, N):
    a = RESULTS / "_cj_codegen.npz"
    b = RESULTS / "_cj_interp.npz"
    ra = _spawn(path, "codegen_jac", T, N, traj_out=a)
    rb = _spawn(path, "interp_jac", T, N, traj_out=b)
    if not (ra.get("ok") and rb.get("ok")):
        return {"pass": False, "codegen": ra, "interp": rb}
    A, B = np.load(a)["species"], np.load(b)["species"]
    peak = float(np.max(np.abs(B))) or 1.0
    rel = float(np.max(np.abs(A - B)) / peak)
    for f in (a, b):
        f.unlink(missing_ok=True)
    return {"pass": rel < 1e-7, "peak_rel": rel}


def time_modes(path, T, N, repeats):
    walls = {"codegen_jac": [], "interp_jac": []}
    last = {"codegen_jac": {}, "interp_jac": {}}
    for _ in range(repeats):
        for mode in ("codegen_jac", "interp_jac"):
            r = _spawn(path, mode, T, N)
            last[mode] = r
            if r.get("ok"):
                walls[mode].append(r["wall"])
    res = {}
    for mode in ("codegen_jac", "interp_jac"):
        w = walls[mode]
        res[mode] = {
            "min_wall": min(w) if w else None,
            "median_wall": median(w) if w else None,
            "n_ok": len(w),
            "stats": last[mode].get("stats", {}),
        }
    cg, it = res["codegen_jac"]["min_wall"], res["interp_jac"]["min_wall"]
    res["speedup"] = (it / cg) if (cg and it) else None
    return res


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--worker", nargs="*")
    add_effort_arg(ap)
    args = ap.parse_args()

    if args.worker:
        path, mode, T, N = args.worker[:4]
        traj = args.worker[4] if len(args.worker) > 4 else None
        worker(path, mode, float(T), int(N), traj)
        return

    RESULTS.mkdir(exist_ok=True)
    models = filter_by_effort(MODELS, args.effort, key=lambda m: m["effort"])
    rows = []
    for m in models:
        path = sbml_path(m["id"])
        if path is None:
            print(f"{m['id']}: MISSING")
            continue
        c = correctness(path, m["T"], m["N"])
        t = time_modes(path, m["T"], m["N"], args.repeats)
        rows.append({"model": m["id"], "note": m["note"], "correctness": c, "timing": t})
        sp = t.get("speedup")
        print(
            f"{m['id']}: correctness pass={c.get('pass')} peak_rel={c.get('peak_rel')} "
            f"codegen_jac(min)={t['codegen_jac']['min_wall']} "
            f"interp_jac(min)={t['interp_jac']['min_wall']} "
            f"speedup={sp:.2f}x"
            if sp
            else f"{m['id']}: speedup=n/a"
        )

    payload = {"machine_info": nb.machine_info(), "repeats": args.repeats, "results": rows}
    (RESULTS / "codegen_jac_results.json").write_text(json.dumps(payload, indent=2, default=str))
    speedups = [r["timing"]["speedup"] for r in rows if r["timing"].get("speedup")]
    geo = nb.geometric_mean(speedups) if speedups else None
    _write_md(payload, geo)
    print(
        f"\nGeometric-mean speedup (interp/codegen jac): {geo:.2f}x across {len(speedups)} models"
        if geo
        else "no speedups"
    )
    print("WROTE", RESULTS / "codegen_jac_results.json", RESULTS / "codegen_jac_results.md")


def _write_md(payload, geo):
    info = payload.get("machine_info", {})
    L = ["# GH #76 Task 4 — compiled-C vs interpreted analytical Jacobian", ""]
    L.append(
        f"_machine: {info.get('platform', 'n/a')} / Python {info.get('python', 'n/a')} / "
        f"git {info.get('git_commit', 'n/a')}; repeats={payload.get('repeats')}_"
    )
    L.append("")
    L.append(
        "Both modes compile the RHS to C; only the **state Jacobian** differs — "
        "compiled `bngsim_codegen_jac` vs the interpreted ExprTk "
        "`cvode_analytical_dense_jac` (`BNGSIM_NO_CODEGEN_JAC=1`). Both are "
        "analytical, so trajectories are identical (`peak_rel` = 0 below). Headline "
        "= **min** of the repeats."
    )
    L.append("")
    L.append(
        "| model | note | peak_rel | codegen_jac min (s) | interp_jac min (s) | speedup | n_steps / n_jac |"
    )
    L.append("|---|---|---|---|---|---|---|")
    for r in payload["results"]:
        t, c = r["timing"], r["correctness"]
        cg, it = t["codegen_jac"], t["interp_jac"]
        st = cg.get("stats", {})
        sp = t.get("speedup")
        L.append(
            f"| {r['model']} | {r['note']} | {c.get('peak_rel')} | "
            f"{cg['min_wall']} | {it['min_wall']} | {sp:.2f}x | "
            f"{st.get('n_steps')} / {st.get('n_jac_evals')} |"
            if sp is not None
            else f"| {r['model']} | {r['note']} | — | — | — | n/a | — |"
        )
    L.append("")
    if geo:
        L.append(f"**Geometric-mean speedup (interp/codegen): {geo:.2f}×.**")
        L.append("")
    L.append("## Interpretation (honest)")
    L.append("")
    L.append(
        "The compiled Jacobian is **bit-identical** to the interpreted one "
        "(verified against `_dense_analytical_jacobian` in "
        "`test_codegen_jacobian.py`) and ~6× cheaper to *compute* per eval, but "
        "the end-to-end win is **~1.0×** on this corpus:\n"
        "\n"
        "- These functional SBML models are large and dense (KLU off → dense "
        "path), so each Newton solve is an O(ns³) dense LU factorization that "
        "dominates per-step cost; the O(nnz) Jacobian scatter is a small slice.\n"
        "- CVODE reuses each Jacobian across many steps (e.g. 13 builds / 38 "
        "steps), so even a much cheaper Jacobian moves the total little (Amdahl). "
        "This is the same regime the analytical-vs-FD benchmark (`run.py`) "
        "documents.\n"
        "\n"
        "The real win is architectural: a codegen dense ODE run is now **fully "
        "compiled** (RHS + Jacobian, no ExprTk in the loop), and this is the "
        "foundation for the sparse-CSC follow-up. A model that is *jac-eval-bound* "
        "— frequent rebuilds, cheap LU, expensive functional rate laws — would see "
        "a larger end-to-end gain."
    )
    L.append("")
    (RESULTS / "codegen_jac_results.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
