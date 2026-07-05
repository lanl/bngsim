#!/usr/bin/env python3
"""Does the analytical Functional Jacobian improve *solver robustness* (not just
per-step cost)?

An exact Jacobian makes CVODE's Newton iteration converge more reliably than an
approximate (finite-difference / colored-FD) one.  When it matters, that shows
up as fewer **nonlinear convergence failures** (`n_nonlin_conv_fails` -- Newton
gave up and CVODE had to cut the step), fewer **error-test failures**
(`n_err_test_fails` -- step rejected), and hence fewer total **steps** to cover
the same horizon.  This is the stiffness benefit the per-step timing benchmark
does not capture (its 5 hand-picked models happen to take near-identical Newton
paths in both modes).

This sweep scans a stride sample of the rr_parity SBML corpus, keeps the models
that (a) have Functional reactions and (b) genuinely attach the analytical
Jacobian, runs each in both modes at DEFAULT tolerance, and compares the
robustness counters.  It reports the distribution and the models with the
largest analytical-favouring (and FD-favouring) gaps -- honestly, including a
null result if the corpus shows no material difference.

Usage::

    python robustness_sweep.py [--stride N] [--max-models M] [--T 100] [--N 50]
    python robustness_sweep.py --worker <sbml_path> <analytical|fd> <T> <N>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODELS_DIR = ROOT / "parity_checks" / "rr_parity" / "models"
RESULTS = HERE / "results"

ROBUSTNESS_KEYS = (
    "n_steps",
    "n_err_test_fails",
    "n_nonlin_conv_fails",
    "n_nonlin_iters",
    "n_jac_evals",
    "n_rhs_evals",
)


def sbml_path(model_id: str) -> Path | None:
    d = MODELS_DIR / model_id
    if not d.is_dir():
        return None
    cand = d / f"{model_id}.xml"
    if cand.exists():
        return cand
    xmls = sorted(d.glob("*.xml"))
    return xmls[0] if xmls else None


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def worker(path, mode, T, N):
    os.environ["BNGSIM_ANALYTICAL_FUNCTIONAL_JAC"] = "1" if mode == "analytical" else "0"
    rec = {"mode": mode}
    try:
        import bngsim
        from bngsim import _jacobian

        model = bngsim.Model.from_sbml(str(path))
        core = model._core
        ctx = core.functional_jacobian_context()
        rec["n_functional_rxns"] = len(ctx.get("functional_reactions") or [])
        # In analytical mode, record whether the analytical Jacobian truly attaches.
        if mode == "analytical":
            # attach already ran inside from_sbml; re-running is idempotent and
            # returns the same verdict without changing the committed state.
            rec["attached"] = bool(_jacobian.attach_functional_jacobian(core))
        sim = bngsim.Simulator(model, method="ode")
        res = sim.run(t_span=(0.0, T), n_points=N)
        ss = dict(getattr(res, "solver_stats", {}) or {})
        rec["stats"] = {k: ss.get(k) for k in ROBUSTNESS_KEYS}
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        rec["error"] = repr(e)
    print(json.dumps(rec))


def _spawn(path, mode, T, N, timeout=120):
    cmd = [
        sys.executable,
        str(HERE / "robustness_sweep.py"),
        "--worker",
        str(path),
        mode,
        repr(T),
        str(N),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "ok": False}
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()]
    if lines:
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            pass
    return {"error": "no_json", "ok": False, "stderr": proc.stderr[-400:]}


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def sample_models(stride, max_models):
    dirs = sorted(p.name for p in MODELS_DIR.iterdir() if p.is_dir())
    picked = dirs[::stride][:max_models]
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=18)
    ap.add_argument("--max-models", type=int, default=70)
    ap.add_argument("--T", type=float, default=100.0)
    ap.add_argument("--N", type=int, default=50)
    ap.add_argument("--worker", nargs=4, metavar=("PATH", "MODE", "T", "N"))
    args = ap.parse_args()

    if args.worker:
        path, mode, T, N = args.worker
        worker(path, mode, float(T), int(N))
        return

    RESULTS.mkdir(exist_ok=True)
    ids = sample_models(args.stride, args.max_models)
    print(f"scanning {len(ids)} model dirs (stride={args.stride}) ...", flush=True)

    rows = []
    n_functional = 0
    for i, mid in enumerate(ids):
        path = sbml_path(mid)
        if path is None:
            continue
        a = _spawn(path, "analytical", args.T, args.N)
        if not a.get("ok"):
            continue
        if not a.get("n_functional_rxns"):
            continue  # Elementary-only: no Functional reactions to differentiate
        n_functional += 1
        if not a.get("attached"):
            # functional but bailed to FD in "analytical" mode -> both modes are
            # FD; no robustness comparison to make. Record and skip.
            rows.append(
                {"model": mid, "n_functional_rxns": a["n_functional_rxns"], "attached": False}
            )
            print(f"[{i}] {mid}: functional but NOT attached (bailed to FD)")
            continue
        f = _spawn(path, "fd", args.T, args.N)
        if not f.get("ok"):
            continue
        sa, sf = a["stats"], f["stats"]

        def delta(key, sa=sa, sf=sf):
            va, vf = sa.get(key), sf.get(key)
            if va is None or vf is None:
                return None
            return vf - va  # positive => FD worse (analytical more robust)

        row = {
            "model": mid,
            "n_functional_rxns": a["n_functional_rxns"],
            "attached": True,
            "analytical": sa,
            "fd": sf,
            "d_steps": delta("n_steps"),
            "d_err_test_fails": delta("n_err_test_fails"),
            "d_nonlin_conv_fails": delta("n_nonlin_conv_fails"),
            "d_nonlin_iters": delta("n_nonlin_iters"),
        }
        rows.append(row)
        print(
            f"[{i}] {mid}: nfunc={a['n_functional_rxns']} "
            f"steps a/fd={sa.get('n_steps')}/{sf.get('n_steps')} "
            f"convfail a/fd={sa.get('n_nonlin_conv_fails')}/{sf.get('n_nonlin_conv_fails')} "
            f"errfail a/fd={sa.get('n_err_test_fails')}/{sf.get('n_err_test_fails')}"
        )

    compared = [r for r in rows if r.get("attached") and "analytical" in r]
    payload = {
        "T": args.T,
        "N": args.N,
        "stride": args.stride,
        "n_scanned": len(ids),
        "n_functional": n_functional,
        "n_compared": len(compared),
        "rows": rows,
    }
    (RESULTS / "robustness_sweep.json").write_text(json.dumps(payload, indent=2))
    _summarize(payload)
    print("WROTE", RESULTS / "robustness_sweep.json")


def _summarize(payload):
    compared = [r for r in payload["rows"] if r.get("attached") and "analytical" in r]
    if not compared:
        print("\nNo attaching functional models in the sample.")
        return

    def total(key):
        return sum(r[key] for r in compared if r.get(key) is not None)

    print("\n=== ROBUSTNESS SUMMARY (positive delta => analytical more robust) ===")
    print(f"compared models: {len(compared)}")
    for key in ("d_steps", "d_err_test_fails", "d_nonlin_conv_fails", "d_nonlin_iters"):
        vals = [r[key] for r in compared if r.get(key) is not None]
        pos = sum(1 for v in vals if v > 0)
        neg = sum(1 for v in vals if v < 0)
        print(
            f"  {key}: total={sum(vals)}  models analytical-better={pos} "
            f"FD-better={neg} tied={len(vals) - pos - neg}"
        )
    # Biggest analytical-favouring gaps by convergence failures then steps.
    ranked = sorted(
        compared,
        key=lambda r: (r.get("d_nonlin_conv_fails") or 0, r.get("d_steps") or 0),
        reverse=True,
    )
    print("\n  Top analytical-favouring (by conv-fail then steps):")
    for r in ranked[:8]:
        print(
            f"    {r['model']}: d_convfail={r.get('d_nonlin_conv_fails')} "
            f"d_steps={r.get('d_steps')} d_errfail={r.get('d_err_test_fails')} "
            f"(steps a/fd={r['analytical'].get('n_steps')}/{r['fd'].get('n_steps')})"
        )


if __name__ == "__main__":
    main()
