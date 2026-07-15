#!/usr/bin/env python3
"""Jacobian-based characterization of the bng_parity ODE test set.

For each ODE model in the corpus we compute two intrinsic, model-level quantities
that predict which linear-algebra regime an implicit stiff solve falls into:

  * **Structural Jacobian density** = nnz(J) / N^2, N = number of ODEs (species).
    nnz is the STRUCTURAL sparsity pattern (union of |J_ij| > 0 over several random
    strictly-positive states), so accidental numeric zeros at one state don't
    deflate it. This is what a sparse linear solver actually sees.

  * **Stiffness ratio** = max|Re lambda| / min_nonzero|Re lambda| over the eigenvalues
    of the REDUCED Jacobian (the conserved-moiety modes removed via BNGsim's own
    ``conservation_laws``). Evaluated at several points along the trajectory;
    we report the MAX over time. Oscillatory models (a weakly-damped complex mode
    would otherwise dominate the denominator) are DETECTED and moved to their own
    category rather than reported as spuriously stiff.

The intent (see the paper supplement) is to partition the ODE test set into the
~O(N) regime (non-stiff and/or non-dense -> explicit or sparse solve) and the
~O(N^3) regime (stiff AND dense -> dense LU dominates). The partition + the
cost~N regression that validates it live in ``--analyze`` mode, which joins this
report with ``runs/report_ode.json`` (per-model N and warm integration cost).

Environment: run with the bngsim editable checkout venv, e.g.
    BNGPATH=/path/to/BioNetGen-2.9.3 \
      ~/Code/bngsim/.venv/bin/python jacobian_characterization.py --limit 5

Reads the vendored corpus (``models/<model_id>``) + ``jobs.json`` read-only; the
only write is the output report JSON. Network generation shells out to BNG2.pl
exactly as the parity runner does (``_bng_common.generate_network``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import _bng_common as bc  # noqa: E402  (sibling module; path injected above)

# ---- tunables (documented; overridable via CLI) ----------------------------
DENSITY_SAMPLES = 5          # random states unioned for the structural pattern
NONZERO_REL_TOL = 1e-9       # |Re lambda| below this * max is treated as zero/marginal
OSC_DAMPING_CUT = 0.01       # complex mode with |Re|/|Im| < this is "oscillatory"
OSC_NEARZERO_BAND = 1e-3     # ...and only if its |Re| sits in the near-zero band
FULL_GRID_MAX_N = 300        # N <= this: evaluate at every trajectory point
DENSE_TIME_SAMPLES = 64      # N >  this: this many log-spaced trajectory points (was 3)
EIG_MAX_N = 5000             # N > this: skip eigen-work (dense eig too costly); density only
DEFAULT_TIMEOUT = 240.0
RNG_SEED = 12345


# ---------------------------------------------------------------------------
# model enumeration + horizon (from the existing parity artifacts)
# ---------------------------------------------------------------------------
def load_ode_jobs() -> list[dict]:
    """ODE jobs from jobs.json (one per vendored model, method == 'ode')."""
    jobs = json.loads((HERE / "jobs.json").read_text())["jobs"]
    return [j for j in jobs if j.get("method") == "ode"]


def load_horizons() -> dict[str, dict]:
    """model_id -> {t_start,t_end,n_steps,rtol,atol,n_species,cost_sec} from report_ode.json."""
    out: dict[str, dict] = {}
    rp = HERE / "runs" / "report_ode.json"
    if not rp.exists():
        return out
    for r in json.loads(rp.read_text()).get("results", []):
        t = r.get("timing") or {}
        spec = t.get("spec") or {}
        bs = t.get("bngsim") or {}
        out[r["model_id"]] = {
            "t_start": spec.get("t_start", 0.0),
            "t_end": spec.get("t_end"),
            "n_steps": spec.get("n_steps"),
            "rtol": spec.get("rtol"),
            "atol": spec.get("atol"),
            "n_species": spec.get("n_species"),
            "cost_sec": bs.get("integrate_warm_median_sec"),
            "outcome": r.get("outcome"),
            "linear_solver": (bs.get("config") or {}).get("linear_solver"),
        }
    return out


# ---------------------------------------------------------------------------
# per-model characterization
# ---------------------------------------------------------------------------
def _prop(obj, name):
    v = getattr(obj, name)
    return v() if callable(v) else v


def _link_matrix(cl: dict, n: int) -> tuple[list[int], np.ndarray]:
    """Build (independent indices, link matrix L: n x n_ind) from conservation laws.

    x = L @ x_ind + const, so the reduced Jacobian is J[ind, :] @ L and its
    eigenvalues are exactly the nonzero eigenvalues of the full Jacobian.
    """
    ind = list(cl["independent"])
    dep = list(cl["dependent"])
    if not ind:  # no conservation laws (n_laws==0): the whole system is independent.
        return list(range(n)), np.eye(n)   # else the reduced Jacobian is empty -> false degenerate
    C = np.asarray(cl["coefficients"], float) if cl.get("coefficients") else np.zeros((0, n))
    n_ind = len(ind)
    L = np.zeros((n, n_ind))
    for b, s in enumerate(ind):
        L[s, b] = 1.0
    for k, d in enumerate(dep):
        cdd = C[k, d]
        if cdd == 0:
            continue
        for b, s in enumerate(ind):
            L[d, b] = -C[k, s] / cdd
    return ind, L


def _time_indices(times: np.ndarray, n: int, k: int = DENSE_TIME_SAMPLES) -> list[int]:
    """Trajectory sample indices for the stiffness sweep.

    For N <= FULL_GRID_MAX_N we evaluate at every output time (the eig is cheap).
    For N > FULL_GRID_MAX_N the full output grid would mean one dense eig per
    output time (~1000 at N~1600), so instead we take up to ``k`` (=DENSE_TIME_SAMPLES)
    log-spaced-in-time points spanning [first t>0, t_final], always pinning index 0
    (the initial time) and the final index. Log spacing concentrates samples where
    fast stiff transients live (early times) while still reaching the final state.
    3 points (the old behavior) under-samples the trajectory max/median badly; ~64
    resolves them well while keeping the eigen-work bounded. Snapping log-spaced
    target times to nearest grid points on a coarse early grid may yield slightly
    fewer than ``k`` distinct indices, which is fine.
    """
    T = np.asarray(times, float)
    m = len(T)
    if n <= FULL_GRID_MAX_N or m <= k:
        return list(range(m))
    pos = T[T > 0]
    t_lo = pos.min() if pos.size else (T[1] if m > 1 else T[0])
    t_hi = T[-1]
    targets = np.geomspace(max(t_lo, 1e-300), max(t_hi, t_lo), num=k)
    picks = {0, m - 1}
    picks.update(int(np.argmin(np.abs(T - tt))) for tt in targets)
    return sorted(picks)


def _classify_eigs(eigs: np.ndarray) -> dict:
    """One time point: max|Re|, min-nonzero|Re|, ratio, and oscillatory flag."""
    re = np.abs(eigs.real)
    im = np.abs(eigs.imag)
    mag = np.abs(eigs)  # sqrt(Re^2 + Im^2)
    max_re = float(re.max()) if re.size else 0.0
    if max_re == 0.0:
        return {"ratio": float("inf"), "max_re": 0.0, "min_re": 0.0, "oscillatory": False}
    floor = NONZERO_REL_TOL * max_re
    # Oscillatory: a *genuine* (magnitude above the zero floor -> not a numerical/
    # conservation-residual zero) weakly-damped complex mode in the near-zero |Re| band.
    # The magnitude floor is essential: without it, machine-zero eigenvalues with a tiny
    # imaginary part (|Re|~1e-13, |Im|~1e-7) are mislabeled oscillatory (e.g. fceri_fyn).
    band = OSC_NEARZERO_BAND * max_re
    osc = bool(np.any(
        (im > 0) & (mag > floor) & (re < band)
        & (re < OSC_DAMPING_CUT * np.maximum(im, 1e-300))
    ))
    nz = re[re > floor]
    ratio = float(max_re / nz.min()) if nz.size else float("inf")
    return {"ratio": ratio, "max_re": max_re, "min_re": (float(nz.min()) if nz.size else 0.0),
            "oscillatory": osc}


def characterize_model(model_id: str, horizon: dict, bng2_pl: str,
                       timeout: float = DEFAULT_TIMEOUT,
                       dense_time_samples: int = DENSE_TIME_SAMPLES) -> dict:
    """Full characterization of one ODE model. Never raises: errors -> status field."""
    import bngsim
    from bngsim import Model, Simulator

    row: dict = {"model_id": model_id, "status": "ok"}
    bngl_path = HERE / "models" / model_id
    if not bngl_path.exists():
        return {**row, "status": "no_bngl"}
    bngl_text = bngl_path.read_text(errors="replace")

    # network generation — same prefix the parity runner uses (single-phase state).
    gen_network = bc._model_gen_network(bngl_text)
    try:
        state_prefix, info = bc.state_setup_prefix(bngl_text, track="ode")
        dirty = bool(info.get("dirty_carryover"))
    except Exception:
        state_prefix, dirty = "", False
    workdir = Path(tempfile.mkdtemp(prefix="bng_jac_"))
    try:
        net_path, netgen_sec, netgen_err = bc.generate_network(
            bngl_text, bng2_pl, workdir, timeout=timeout,
            gen_network=gen_network, state_prefix=("" if dirty else state_prefix),
        )
        if net_path is None:
            return {**row, "status": "netgen_failed", "detail": netgen_err}

        m = Model.from_net(str(net_path))
        core = m._core
        n = int(_prop(m, "n_species"))
        row["N"] = n
        row["n_reactions"] = int(_prop(m, "n_reactions"))
        cl = core.conservation_laws
        row["n_conservation_laws"] = int(cl["n_laws"])
        row["rank"] = n - int(cl["n_laws"])
        row["dirty_carryover"] = dirty

        # BNGsim's own codegen'd analytical Jacobian (exact; what the integrator uses).
        # core._dense_analytical_jacobian(t, conc) -> flat column-major; reshape order="F".
        m.prepare_analytical_jacobian()
        row["analytical_jacobian_complete"] = bool(
            getattr(core, "analytical_jacobian_complete", False))
        row["jacobian_method"] = "native_analytical"

        def Jat(y, t=0.0):
            flat = core._dense_analytical_jacobian(float(t), [float(v) for v in y])
            return np.asarray(flat, float).reshape(n, n, order="F")

        # structural density: union of nonzeros over random positive states. The native
        # Jacobian returns EXACT zeros where structurally zero, so |J|>0 is the pattern.
        rng = np.random.default_rng(RNG_SEED)
        pat = np.zeros((n, n), bool)
        for _ in range(DENSITY_SAMPLES):
            pat |= np.abs(Jat(rng.uniform(0.1, 2.0, n))) > 0
        row["nnz"] = int(pat.sum())
        row["density"] = row["nnz"] / (n * n)

        if n > EIG_MAX_N:
            return {**row, "status": "ok_density_only",
                    "detail": f"N={n} > EIG_MAX_N={EIG_MAX_N}; stiffness skipped"}

        # trajectory (match the parity horizon where available)
        t_start = horizon.get("t_start", 0.0) or 0.0
        t_end = horizon.get("t_end") or 100.0
        n_steps = horizon.get("n_steps") or 100
        run_kw = {}
        if horizon.get("rtol"):
            run_kw["rtol"] = horizon["rtol"]
        if horizon.get("atol"):
            run_kw["atol"] = horizon["atol"]
        res = Simulator(m, method="ode").run(
            t_span=(float(t_start), float(t_end)), n_points=int(n_steps) + 1, **run_kw)
        X = np.asarray(res.species, float)
        T = np.asarray(res.time, float)

        ind, L = _link_matrix(cl, n)
        idxs = _time_indices(T, n, dense_time_samples)
        per_time = []
        any_osc = False
        for i in idxs:
            J = Jat(X[i], T[i])
            pat |= np.abs(J) > 0  # amortize structural pattern
            Jred = J[np.ix_(ind, range(n))] @ L
            eigs = np.linalg.eigvals(Jred)
            c = _classify_eigs(eigs)
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
        row["category"] = "oscillatory" if any_osc else "pending"  # stiff/nonstiff set in --analyze
        return row
    except Exception as exc:
        return {**row, "status": "error",
                "detail": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()[-1500:]}
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# analysis: partition into O(N) vs O(N^3) regimes + validate by cost~N regression
# ---------------------------------------------------------------------------
def _loglog_fit(N, cost):
    """Slope/intercept/R^2 of log10(cost) ~ log10(N)."""
    N = np.asarray(N, float); cost = np.asarray(cost, float)
    ok = (N > 0) & (cost > 0)
    x = np.log10(N[ok]); y = np.log10(cost[ok])
    if x.size < 3 or np.ptp(x) == 0:
        return {"slope": None, "intercept": None, "r2": None, "n": int(x.size)}
    A = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ np.array([slope, intercept])
    ss_res = float(np.sum((y - yhat) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2, "n": int(x.size)}


def analyze(char_path: Path, dense_threshold: float | None,
            stiff_threshold: float | None) -> dict:
    """Reclassify, partition by solver-relevant regime (sparse/dense stiff), regress cost~N."""
    char = json.loads(Path(char_path).read_text())["results"]
    horizons = load_horizons()

    def maxre(r):
        return max([p.get("max_re", 0) for p in (r.get("per_time") or [])], default=0.0)

    pts = []
    for r in char:
        if not str(r.get("status", "")).startswith("ok"):
            continue
        N = r.get("N"); dens = r.get("density")
        if N is None or dens is None:
            continue
        h = horizons.get(r["model_id"], {})
        degenerate = (dens == 0) or (maxre(r) == 0)   # zero-Jacobian / trivial
        pts.append({"model_id": r["model_id"], "N": N, "density": dens,
                    "stiffness": r.get("stiffness_ratio_max"), "degenerate": degenerate,
                    "oscillatory": bool(r.get("oscillatory")) and not degenerate,
                    "cost_sec": h.get("cost_sec"), "linear_solver": h.get("linear_solver")})

    deg = [p for p in pts if p["degenerate"]]
    osc = [p for p in pts if p["oscillatory"]]
    live = [p for p in pts if not p["degenerate"] and not p["oscillatory"]
            and p["stiffness"] is not None and np.isfinite(p["stiffness"])]

    dvals = np.array([p["density"] for p in live]); svals = np.array([p["stiffness"] for p in live])
    Nvals = np.array([p["N"] for p in live], float)

    def pct(a, q):
        return float(np.percentile(a, q)) if a.size else None
    corr = float(np.corrcoef(np.log10(Nvals), dvals)[0, 1]) if len(live) > 2 else None

    dth = dense_threshold if dense_threshold is not None else float(np.median(dvals))
    sth = stiff_threshold if stiff_threshold is not None else 1e3

    # Regime is model x SOLVER: among stiff models (where the linear solve dominates),
    # sparse -> a sparse-aware solver (KLU) stays ~O(N) while dense-only tools pay O(N^3);
    # dense -> everyone pays O(N^3). Non-stiff -> explicit-ish, O(N) for all.
    for p in live:
        p["cls"] = ("nonstiff" if p["stiffness"] < sth
                    else ("sparse_stiff" if p["density"] < dth else "dense_stiff"))

    def fit(group):
        c = [p for p in group if p.get("cost_sec")]
        return _loglog_fit([p["N"] for p in c], [p["cost_sec"] for p in c])

    classes = {}
    for c in ("sparse_stiff", "dense_stiff", "nonstiff"):
        g = [p for p in live if p["cls"] == c]
        Ns = np.array([p["N"] for p in g])
        solv = {}
        for p in g:
            solv[p["linear_solver"]] = solv.get(p["linear_solver"], 0) + 1
        classes[c] = {"n": len(g),
                      "N_min": int(Ns.min()) if g else None,
                      "N_med": int(np.median(Ns)) if g else None,
                      "N_max": int(Ns.max()) if g else None,
                      "solvers": solv, "cost_vs_N": fit(g)}

    ladder = sorted([p for p in live if p["cls"] == "sparse_stiff"], key=lambda p: -p["N"])
    ladder_rows = [{"model_id": p["model_id"], "N": p["N"], "density": round(p["density"], 3),
                    "stiffness": p["stiffness"], "solver": p["linear_solver"]} for p in ladder]

    print("\n===== Jacobian regime analysis (reframed: solver x N) =====")
    print(f"ok {len(pts)} -> degenerate {len(deg)}, genuine oscillatory {len(osc)}, live {len(live)}")
    print(f"corr(log10 N, density) = {corr:+.2f}  (negative => big networks are sparse)")
    print(f"density median {np.median(dvals):.3f} | thresholds: dense>= {dth:.3f}, stiff>= {sth:g}")
    for c in ("sparse_stiff", "dense_stiff", "nonstiff"):
        cc = classes[c]; f = cc["cost_vs_N"]
        sl = "n/a" if f["slope"] is None else f"N^{f['slope']:.2f} (R^2={f['r2']:.2f})"
        print(f"  {c:13s} n={cc['n']:3d}  N[min/med/max]={cc['N_min']}/{cc['N_med']}/{cc['N_max']}"
              f"  cost~{sl}  solvers={cc['solvers']}")
    print("\n  sparse-stiff ladder (BNGsim-KLU-advantage candidates), top by N:")
    for row in ladder_rows[:15]:
        print(f"    N={row['N']:4d} dens={row['density']:.3f} stiff={row['stiffness']:.1g} "
              f"solv={row['solver']}  {row['model_id'].split('/')[-1]}")

    summary = {
        "counts": {"ok": len(pts), "degenerate": len(deg), "oscillatory": len(osc), "live": len(live)},
        "corr_logN_density": corr,
        "thresholds": {"dense>=": dth, "stiff>=": sth},
        "density_pctiles": {q: pct(dvals, q) for q in (10, 25, 50, 75, 90)},
        "stiffness_log10_pctiles": {q: pct(np.log10(svals), q) for q in (10, 25, 50, 75, 90)},
        "classes": classes,
    }
    return {"summary": summary, "sparse_stiff_ladder": ladder_rows,
            "groups": {c: [p["model_id"] for p in live if p["cls"] == c]
                       for c in ("sparse_stiff", "dense_stiff", "nonstiff")},
            "degenerate": [p["model_id"] for p in deg],
            "oscillatory": [p["model_id"] for p in osc]}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "runs" / "jacobian_characterization.json")
    ap.add_argument("--limit", type=int, default=None, help="characterize only the first N models")
    ap.add_argument("--model", type=str, default=None, help="substring filter on model_id")
    ap.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    ap.add_argument("--dense-time-samples", type=int, default=DENSE_TIME_SAMPLES,
                    help="for N > FULL_GRID_MAX_N: number of log-spaced trajectory "
                         "points at which stiffness is evaluated (default: %(default)s)")
    ap.add_argument("--max-n", type=int, default=None,
                    help="optional: skip models whose report n_species exceeds this "
                         "(the native analytical Jacobian handles large N fine; default: no skip)")
    ap.add_argument("--analyze", nargs="?", const=True, default=False,
                    help="analysis mode: partition + cost~N regression from a "
                         "characterization JSON (default: --out path)")
    ap.add_argument("--dense-threshold", type=float, default=None)
    ap.add_argument("--stiff-threshold", type=float, default=None)
    args = ap.parse_args()

    if args.analyze:
        path = args.out if args.analyze is True else Path(args.analyze)
        res = analyze(path, args.dense_threshold, args.stiff_threshold)
        ap_out = path.with_name(path.stem + "_analysis.json")
        ap_out.write_text(json.dumps(res, indent=1))
        print(f"[jac] wrote {ap_out}")
        return 0

    bng2_pl = bc.resolve_bng2_pl(os.environ.get("BNGPATH") or os.environ.get("BNG2_PL"))
    jobs = load_ode_jobs()
    horizons = load_horizons()
    ids = [j["model_id"] for j in jobs]
    if args.model:
        ids = [i for i in ids if args.model in i]
    if args.limit:
        ids = ids[: args.limit]

    print(f"[jac] {len(ids)} ODE models | bng2_pl={bng2_pl}", flush=True)
    rows = []
    t0 = time.perf_counter()
    for k, mid in enumerate(ids, 1):
        h = horizons.get(mid, {})
        if args.max_n is not None and (h.get("n_species") or 0) > args.max_n:
            rows.append({"model_id": mid, "status": "skipped_too_large",
                         "N": h.get("n_species")})
            print(f"[{k:3d}/{len(ids)}] skipped_too_large  N={h.get('n_species')} {mid}",
                  flush=True)
            continue
        r = characterize_model(mid, h, bng2_pl, timeout=args.timeout,
                               dense_time_samples=args.dense_time_samples)
        rows.append(r)
        tag = r.get("status")
        extra = ""
        if r.get("N") is not None:
            extra = (f"N={r['N']} dens={r.get('density', float('nan')):.3f} "
                     f"stiff[max/med]={r.get('stiffness_ratio_max', float('nan')):.3g}/"
                     f"{r.get('stiffness_ratio_median', float('nan')):.3g} "
                     f"npts={r.get('n_time_points', '-')} "
                     f"{'OSC ' if r.get('oscillatory') else ''}")
        print(f"[{k:3d}/{len(ids)}] {tag:16s} {extra}{mid}", flush=True)

    out = {
        "_meta": {
            "generator": "jacobian_characterization.py",
            "bngsim_version": __import__("bngsim").__version__,
            "n_models": len(rows),
            "params": {
                "DENSITY_SAMPLES": DENSITY_SAMPLES, "NONZERO_REL_TOL": NONZERO_REL_TOL,
                "OSC_DAMPING_CUT": OSC_DAMPING_CUT, "OSC_NEARZERO_BAND": OSC_NEARZERO_BAND,
                "FULL_GRID_MAX_N": FULL_GRID_MAX_N,
                "DENSE_TIME_SAMPLES": args.dense_time_samples, "EIG_MAX_N": EIG_MAX_N,
            },
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        },
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    ok = sum(1 for r in rows if str(r.get("status")).startswith("ok"))
    print(f"[jac] wrote {args.out}  ({ok}/{len(rows)} characterized, "
          f"{out['_meta']['elapsed_sec']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
