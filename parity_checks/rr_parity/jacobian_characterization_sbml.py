#!/usr/bin/env python3
"""Jacobian characterization of the rr_parity SBML BioModels ODE corpus.

Same metrics and regime classification as
``bng_parity/jacobian_characterization.py`` (structural Jacobian density = nnz/N^2;
stiffness ratio = max|Re lambda| / min-nonzero|Re lambda| over the conservation-reduced
Jacobian, maximized along the trajectory; oscillatory / degenerate handling), but the
model is loaded from SBML via ``Model.from_sbml`` instead of a BNG2.pl-generated ``.net``
-- so there is no network-generation step. The metric helpers are imported from the
bng_parity module so the two corpora are characterized by identical code.

Environment: the bngsim editable checkout venv, e.g.
    ~/Code/bngsim/.venv/bin/python jacobian_characterization_sbml.py --limit 5

Reads ``ode_jobs.json`` + the vendored ``models/<id>/*.xml`` read-only; the only write
is the output report JSON (default ``runs/jacobian_characterization_sbml.json``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BNG_PARITY = HERE.parent / "bng_parity"
if str(BNG_PARITY) not in sys.path:
    sys.path.insert(0, str(BNG_PARITY))
import jacobian_characterization as jc  # noqa: E402  (shared metric helpers)

INTEGRATION_TIMEOUT = 7.0  # wall-clock cap per model's ODE solve (bounds stiff BioModels;
# BNGsim may retry once with an FD Jacobian, so worst case ~2x)


def load_ode_jobs() -> list[dict]:
    jobs = json.loads((HERE / "ode_jobs.json").read_text())["jobs"]
    return [j for j in jobs if j.get("method") == "ode"]


def characterize_sbml(model_id: str, sbml_rel: str, horizon: dict) -> dict:
    """Characterize one SBML model. Never raises: errors -> status field."""
    from bngsim import Model, Simulator

    row: dict = {"model_id": model_id, "status": "ok"}
    path = HERE / sbml_rel
    if not path.exists():
        return {**row, "status": "no_sbml"}
    try:
        m = Model.from_sbml(str(path))
        core = m._core
        n = int(jc._prop(m, "n_species"))
        if n == 0:
            return {**row, "status": "no_species", "N": 0}
        m.prepare_analytical_jacobian()
        row["analytical_jacobian_complete"] = bool(
            getattr(core, "analytical_jacobian_complete", False)
        )
        row["jacobian_method"] = "native_analytical"
        row["N"] = n
        row["n_reactions"] = int(jc._prop(m, "n_reactions"))
        cl = core.conservation_laws
        row["n_conservation_laws"] = int(cl["n_laws"])
        row["rank"] = n - int(cl["n_laws"])

        def Jat(y, t=0.0):
            flat = core._dense_analytical_jacobian(float(t), [float(v) for v in y])
            return np.asarray(flat, float).reshape(n, n, order="F")

        rng = np.random.default_rng(jc.RNG_SEED)
        pat = np.zeros((n, n), bool)
        for _ in range(jc.DENSITY_SAMPLES):
            pat |= np.abs(Jat(rng.uniform(0.1, 2.0, n))) > 0
        row["nnz"] = int(pat.sum())
        row["density"] = row["nnz"] / (n * n)

        if n > jc.EIG_MAX_N:
            return {
                **row,
                "status": "ok_density_only",
                "detail": f"N={n} > EIG_MAX_N; stiffness skipped",
            }
    except Exception as exc:
        import traceback

        return {
            **row,
            "status": "error",
            "detail": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-1200:],
        }

    # Trajectory + stiffness in a SEPARATE try, with a wall-clock integration cap:
    # a pathologically stiff BioModel that CVODE cannot integrate must neither stall
    # the sweep nor lose its already-computed density.
    try:
        t_start = horizon.get("t_start", 0.0) or 0.0
        t_end = horizon.get("t_end") or 100.0
        n_pts = horizon.get("n_points") or 101
        run_kw = {"timeout": INTEGRATION_TIMEOUT}
        if horizon.get("rtol"):
            run_kw["rtol"] = horizon["rtol"]
        if horizon.get("atol"):
            run_kw["atol"] = horizon["atol"]
        res = Simulator(m, method="ode").run(
            t_span=(float(t_start), float(t_end)), n_points=int(n_pts), **run_kw
        )
        X = np.asarray(res.species, float)
        T = np.asarray(res.time, float)

        ind, L = jc._link_matrix(cl, n)
        idxs = jc._time_indices(T, n)
        per_time = []
        any_osc = False
        for i in idxs:
            J = Jat(X[i], T[i])
            pat |= np.abs(J) > 0
            eigs = np.linalg.eigvals(J[np.ix_(ind, range(n))] @ L)
            c = jc._classify_eigs(eigs)
            any_osc = any_osc or c["oscillatory"]
            per_time.append({"t": float(T[i]), **c})
        row["nnz"] = int(pat.sum())
        row["density"] = row["nnz"] / (n * n)
        finite = [p["ratio"] for p in per_time if np.isfinite(p["ratio"])]
        row["stiffness_ratio_max"] = float(max(finite)) if finite else float("inf")
        # median over finite per-time ratios = the "sustained" stiffness (vs the peak).
        row["stiffness_ratio_median"] = float(np.median(finite)) if finite else float("inf")
        row["n_time_points"] = len(idxs)
        row["oscillatory"] = bool(any_osc)
        row["per_time"] = per_time
        return row
    except Exception as exc:
        # keep density (already in row); just no stiffness classification
        row["status"] = "ok_no_stiffness"
        row["detail"] = f"integration failed: {type(exc).__name__}: {exc}"[:200]
        return row


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out", type=Path, default=HERE / "runs" / "jacobian_characterization_sbml.json"
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", type=str, default=None)
    args = ap.parse_args()

    jobs = load_ode_jobs()
    if args.model:
        jobs = [j for j in jobs if args.model in j["model_id"]]
    if args.limit:
        jobs = jobs[: args.limit]

    print(f"[jac-sbml] {len(jobs)} SBML ODE models", flush=True)
    rows = []
    t0 = time.perf_counter()
    for k, j in enumerate(jobs, 1):
        r = characterize_sbml(j["model_id"], j["model"], j.get("params", {}))
        rows.append(r)
        extra = ""
        if r.get("N") is not None:
            extra = (
                f"N={r['N']} dens={r.get('density', float('nan')):.3f} "
                f"stiff[max/med]={r.get('stiffness_ratio_max', float('nan')):.3g}/"
                f"{r.get('stiffness_ratio_median', float('nan')):.3g} "
                f"{'OSC ' if r.get('oscillatory') else ''}"
            )
        print(
            f"[{k:4d}/{len(jobs)}] {str(r.get('status')):16s} {extra}{j['model_id']}", flush=True
        )

    out = {
        "_meta": {
            "generator": "jacobian_characterization_sbml.py",
            "bngsim_version": __import__("bngsim").__version__,
            "n_models": len(rows),
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        },
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    ok = sum(1 for r in rows if str(r.get("status")).startswith("ok"))
    print(
        f"[jac-sbml] wrote {args.out} ({ok}/{len(rows)} characterized, "
        f"{out['_meta']['elapsed_sec']}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
