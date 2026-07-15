#!/usr/bin/env python3
"""Recovery/robustness: re-time a model's AMICI KLU and dense variants in SEPARATE
fork-isolated children off the (already-cached) compile, then patch its entry in
report_ode_engines_s3.json.

Why: the main harness times KLU+dense in ONE child, so a slow/near-intractable dense
solve at large N (fceri_fyn, N=1281) can hit the child's wall timeout and lose the
(fast, important) KLU result with it. Splitting guarantees KLU is preserved even if
dense times out.

    export BNGPATH=~/Simulations/BioNetGen-2.9.3
    ~/Code/PyBNF-Private/bngsim/.venv/bin/python patch_amici_split.py \
        --models fcerifyn --dense-timeout 1800
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import run_s3_timing as H  # the harness module (same dir); reuses run_in_child, paths


def _amici_one(sbml_path, hz, linear_solver):
    """Build (cache hit) + time ONE AMICI linear-solver variant. Returns the same
    normalized dict the harness produces."""
    import amici  # noqa: F401
    import amici.sim.sundials as ss
    sys.path.insert(0, str(H.AMICI_PARITY))
    import _amici_common as amc

    sbml_str = Path(sbml_path).read_text()
    model, bt, cached = amc._build_model(sbml_str)  # cache hit -> load only
    build = (bt["parse_sec"] + bt["interpret_sec"] + bt["jac_derive_sec"]
             + bt["codegen_sec"] + bt["compile_sec"] + bt["load_sec"])
    enumval = ss.LinearSolver_KLU if linear_solver == "KLU" else ss.LinearSolver_dense
    solver = model.create_solver()
    solver.set_relative_tolerance(hz["rtol"])
    solver.set_absolute_tolerance(hz["atol"])
    solver.set_sensitivity_order(getattr(ss, "SensitivityOrder_none", 0))
    solver.set_linear_solver(enumval)
    model.set_timepoints(np.linspace(hz["t_start"], hz["t_end"], hz["n_points"]))

    t1 = time.perf_counter()
    rd = model.simulate(solver=solver)
    cold = time.perf_counter() - t1
    if int(rd.status) != 0:
        return {**H._err(f"AMICI status {rd.status} (cold)"),
                "build_sec": round(build, 6), "build_breakdown": bt}
    warm = []
    for _ in range(H.rc._warm_rep_count(cold)):
        try:
            t1 = time.perf_counter()
            r2 = model.simulate(solver=solver)
            if int(r2.status) != 0:
                break
            warm.append(time.perf_counter() - t1)
        except Exception:
            break
    integ = H.rc._integrate_stats(cold, warm)
    try:
        ls = ss.LinearSolver(solver.get_linear_solver()).name
    except Exception:
        ls = str(solver.get_linear_solver())
    cfg = {"codegen": "C++ (compiled)", "jacobian": "analytical (symbolic)",
           "linear_solver": ls, "cached": cached}
    note = ("AMICI codegen+compile is one-time and shared by amici_klu and amici_dense "
            "(paid once); build_sec repeats it in both rows.")
    d = H._norm(build, bt, cached, integ, cfg, int(np.asarray(rd.x).shape[1]),
                extra={"integrate_cpu_ms": round(float(rd.cpu_time), 4), "_note": note})
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="fcerifyn", help="Comma-separated model key/label filter.")
    ap.add_argument("--klu-timeout", type=float, default=600.0)
    ap.add_argument("--dense-timeout", type=float, default=1800.0)
    ap.add_argument("--skip-klu", action="store_true")
    ap.add_argument("--skip-dense", action="store_true")
    args = ap.parse_args()

    subs = [s.strip() for s in args.models.split(",") if s.strip()]

    def _save(engine, value):
        # Re-read + patch + write, so KLU is persisted BEFORE the slow dense solve.
        doc = json.loads(H.OUT.read_text())
        for rr in doc["results"]:
            if any(s in rr["model_id"] or s in rr["label"] for s in subs):
                rr["engines"][engine] = value
                rr["_complete"] = all(e in rr["engines"] for e in doc["_meta"]["engines"])
        tmp = H.OUT.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=1))
        tmp.replace(H.OUT)

    doc = json.loads(H.OUT.read_text())
    targets = [r for r in doc["results"] if any(s in r["model_id"] or s in r["label"] for s in subs)]
    if not targets:
        print("  no matching models.")
        return 0
    for r in targets:
        san = r["model_id"].replace("/", "__")
        sbml_path = str(H.SBML_DIR / f"{san}.xml")
        if not Path(sbml_path).exists():
            print(f"  {r['label']}: SBML missing ({sbml_path}); skip")
            continue
        hz = r["horizon"]
        if not args.skip_klu:
            print(f"  {r['label']} N={r['n_species']}: KLU child (timeout {args.klu_timeout:g}s) ...", flush=True)
            klu = H.run_in_child(lambda: _amici_one(sbml_path, hz, "KLU"), args.klu_timeout)
            H._print_engine("amici_klu", klu)
            _save("amici_klu", klu)  # persist KLU immediately
        if not args.skip_dense:
            print(f"  {r['label']}: dense child (timeout {args.dense_timeout:g}s) ...", flush=True)
            dense = H.run_in_child(lambda: _amici_one(sbml_path, hz, "dense"), args.dense_timeout)
            H._print_engine("amici_dense", dense)
            _save("amici_dense", dense)
    print(f"  patched: {H.OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
