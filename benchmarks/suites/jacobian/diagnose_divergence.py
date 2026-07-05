#!/usr/bin/env python3
"""Characterize analytical-vs-FD Functional Jacobian trajectory divergence.

The speedup benchmark's correctness gate found that for some models the
analytical and finite-difference (FD) Jacobians produce trajectories that
differ by far more than the "~1e-15" claimed in the GH #76 kickoff.  This
script decides whether that divergence is

  (a) BENIGN -- a different Jacobian changes CVODE's Newton / step-acceptance
      path, so at a *fixed* rtol/atol the two runs land on different but each
      tolerance-valid trajectories.  Then the gap is ~integration tolerance
      and SHRINKS as tolerance tightens (both converge to the same answer); or
  (b) REAL -- the two Jacobians disagree on a significant species, so the gap
      PERSISTS at tight tolerance.

Method: run each (mode in {analytical, fd}) x (tol in {default, tight}) in a
fresh subprocess (the Jacobian choice is a load-time env decision), save the
species trajectory, and report the max abs / peak-relative diff for the key
comparisons.  ``tight`` = rtol 1e-10, atol 1e-12.

Usage::

    python diagnose_divergence.py MODEL9089538076 [T] [N]
    python diagnose_divergence.py --worker <path> <mode> <T> <N> <rtol> <atol> <out.npz>
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODELS_DIR = ROOT / "parity_checks" / "rr_parity" / "models"
TIGHT = (1e-10, 1e-12)


def sbml_path(model_id: str) -> Path | None:
    d = MODELS_DIR / model_id
    if not d.is_dir():
        return None
    cand = d / f"{model_id}.xml"
    if cand.exists():
        return cand
    xmls = sorted(d.glob("*.xml"))
    return xmls[0] if xmls else None


def worker(path, mode, T, N, rtol, atol, out):
    import os
    import time

    os.environ["BNGSIM_ANALYTICAL_FUNCTIONAL_JAC"] = "1" if mode == "analytical" else "0"
    rec = {"mode": mode, "rtol": rtol, "atol": atol}
    try:
        import bngsim

        model = bngsim.Model.from_sbml(str(path))
        sim = bngsim.Simulator(model, method="ode")
        kw = {"t_span": (0.0, T), "n_points": N}
        if rtol > 0:
            kw["rtol"] = rtol
        if atol > 0:
            kw["atol"] = atol
        t0 = time.perf_counter()
        res = sim.run(**kw)
        rec["wall"] = time.perf_counter() - t0
        np.savez(
            out,
            time=np.asarray(res.time, dtype=float),
            species=np.asarray(res.species, dtype=float),
            names=np.asarray(list(res.species_names), dtype=object),
        )
        rec["stats"] = dict(getattr(res, "solver_stats", {}) or {})
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        import traceback

        rec["error"] = repr(e)
        rec["traceback"] = traceback.format_exc()
    print(json.dumps(rec))


def _spawn(path, mode, T, N, rtol, atol, out):
    cmd = [
        sys.executable,
        str(HERE / "diagnose_divergence.py"),
        "--worker",
        str(path),
        mode,
        repr(T),
        str(N),
        repr(rtol),
        repr(atol),
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    if lines:
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            pass
    return {"error": "no_json", "stderr": proc.stderr[-1500:]}


def compare(a_npz, b_npz, label):
    A, B = np.load(a_npz, allow_pickle=True), np.load(b_npz, allow_pickle=True)
    sa, sb = A["species"], B["species"]
    if sa.shape != sb.shape:
        return {"label": label, "shape": f"{sa.shape} vs {sb.shape}"}
    peak = float(np.max(np.abs(sb))) or 1.0
    absdiff = np.abs(sa - sb)
    max_abs = float(np.max(absdiff))
    # peak-relative (the repo's cross_validate convention): tolerance-aware
    peak_rel = max_abs / peak
    # raw element-wise relative, floored, for context
    denom = np.maximum(np.abs(sa), np.abs(sb))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(denom > 1e-12, absdiff / denom, 0.0)
    idx = np.unravel_index(int(np.argmax(absdiff)), absdiff.shape)
    names = A["names"]
    sp_name = str(names[idx[1]]) if idx[1] < len(names) else f"sp{idx[1]}"
    return {
        "label": label,
        "max_abs": max_abs,
        "peak": peak,
        "peak_rel": peak_rel,
        "max_elt_rel": float(np.max(rel)),
        "at": {"t_idx": int(idx[0]), "species": sp_name, "a": float(sa[idx]), "b": float(sb[idx])},
    }


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _, _, path, mode, T, N, rtol, atol, out = sys.argv
        worker(path, mode, float(T), int(N), float(rtol), float(atol), out)
        return

    model_id = sys.argv[1]
    T = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    path = sbml_path(model_id)
    if path is None:
        print(f"MISSING {model_id}")
        return

    d = HERE / "results" / "_diag"
    d.mkdir(parents=True, exist_ok=True)
    runs = {}
    for mode in ("analytical", "fd"):
        for tol_name, (rt, at) in (("default", (0.0, 0.0)), ("tight", TIGHT)):
            out = d / f"{model_id}_{mode}_{tol_name}.npz"
            rec = _spawn(path, mode, T, N, rt, at, out)
            runs[(mode, tol_name)] = {"rec": rec, "npz": out}
            print(
                f"  {mode}/{tol_name}: ok={rec.get('ok')} wall={rec.get('wall')} "
                f"err={rec.get('error')}"
            )

    if not all(v["rec"].get("ok") for v in runs.values()):
        print("Some runs failed; aborting comparison.")
        print(json.dumps({k[0] + "/" + k[1]: v["rec"] for k, v in runs.items()}, indent=2))
        return

    comps = [
        compare(
            runs[("analytical", "default")]["npz"],
            runs[("fd", "default")]["npz"],
            "analytical_default vs fd_default (the speedup-mode pair)",
        ),
        compare(
            runs[("analytical", "tight")]["npz"],
            runs[("fd", "tight")]["npz"],
            "analytical_tight vs fd_tight (do they converge together?)",
        ),
        compare(
            runs[("analytical", "default")]["npz"],
            runs[("analytical", "tight")]["npz"],
            "analytical: default vs tight (is analytical converging?)",
        ),
        compare(
            runs[("fd", "default")]["npz"],
            runs[("fd", "tight")]["npz"],
            "fd: default vs tight (is fd converging?)",
        ),
    ]
    out_json = {
        "model": model_id,
        "T": T,
        "N": N,
        "runs": {
            f"{k[0]}/{k[1]}": {"wall": v["rec"].get("wall"), "stats": v["rec"].get("stats")}
            for k, v in runs.items()
        },
        "comparisons": comps,
    }
    (HERE / "results" / f"diag_{model_id}.json").write_text(json.dumps(out_json, indent=2))
    print()
    for c in comps:
        print(f"* {c['label']}")
        print(
            f"    max_abs={c.get('max_abs'):.3e}  peak={c.get('peak'):.3e}  "
            f"peak_rel={c.get('peak_rel'):.3e}  max_elt_rel={c.get('max_elt_rel'):.3e}"
        )
        at = c.get("at", {})
        print(
            f"    worst: t_idx={at.get('t_idx')} species={at.get('species')} "
            f"a={at.get('a'):.6g} b={at.get('b'):.6g}"
        )
    print(f"\nWROTE results/diag_{model_id}.json")


if __name__ == "__main__":
    main()
