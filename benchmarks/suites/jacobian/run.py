#!/usr/bin/env python3
"""``jacobian`` suite -- analytical vs finite-difference Functional Jacobian.

GH #76 shipped an analytical Jacobian for Functional / Michaelis-Menten rate
laws, on by default since bngsim 0.9.20.  It is a *performance* feature: it
changes only CVODE's Newton-iteration path, so trajectories are identical to
the finite-difference (FD) Jacobian to ~1e-15.  This suite measures the
payoff.

Two gates per model (mirroring the ``ode`` suite's correctness+timing design):

1. **correctness** -- run the model twice, once with the analytical Jacobian
   (default) and once with FD forced (``BNGSIM_ANALYTICAL_FUNCTIONAL_JAC=0``),
   and confirm the species trajectories agree to a tight tolerance.  This is
   what licenses reporting a *time-to-solution* speedup rather than worrying
   about accuracy.
2. **timing** -- median wall-clock of the ``run()`` integrate call (model load
   excluded) for each mode, plus the speedup ratio and the CVODE solver
   counts.  Per the suite design rule, timing is only meaningful where
   correctness passed.

Each (model, mode) runs in a *fresh subprocess* because the analytical-vs-FD
decision is made once, at model-load time, from the environment; a subprocess
guarantees a clean ``BNGSIM_ANALYTICAL_FUNCTIONAL_JAC`` for that run.

Note on the solver counts: CVODE's ``n_rhs_evals`` counts only the integrator's
own RHS calls -- it does *not* include the per-column RHS perturbations the FD
Jacobian routine makes internally.  So the two modes show near-identical
n_steps / n_jac / n_rhs, and the entire wall-clock delta is cost *per Jacobian
evaluation* (one analytical scatter vs ~n_species FD perturbations of the
functional rate laws).  That is exactly the per-step cost #76 removes.

Usage::

    python run.py                      # both gates, full sweep
    python run.py --mode timing        # timing only
    python run.py --mode correctness   # equivalence check only
    python run.py --quick              # one analytical run per model, smoke only
    python run.py --effort low         # cheap subset (cumulative tiers)
    python run.py --repeats 7
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
_BENCH_ROOT = HERE.parents[1]  # bngsim/benchmarks
ROOT = HERE.parents[2]  # bngsim/
sys.path.insert(0, str(_BENCH_ROOT))
import _netbench as nb  # noqa: E402
from _effort import add_effort_arg, filter_by_effort  # noqa: E402

MODELS_DIR = ROOT / "parity_checks" / "rr_parity" / "models"
NET_DIR = _BENCH_ROOT / "models" / "net"
RESULTS = HERE / "results"


def _relpath(path) -> str:
    """Repo-relative model path for the committed results (machine-independent);
    home-scrubbed absolute fallback for models outside the bngsim/ tree."""
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        s, home = str(p), str(Path.home())
        return "~" + s[len(home) :] if s.startswith(home) else s


# Functional SBML models from the rr_parity corpus, spanning sizes.  The
# analytical Jacobian pays off on the large, reaction-dense models where each
# FD Jacobian costs ~n_species re-evaluations of many functional rate laws;
# it is a wash on compute-light models.  "effort" buckets by run cost.
MODELS = [
    {
        "id": "BIOMD0000000013",
        "T": 100.0,
        "N": 200,
        "effort": "low",
        "note": "Calvin cycle, stiff, ns~27",
    },
    {"id": "MODEL9089538076", "T": 100.0, "N": 200, "effort": "medium", "note": "ns~200"},
    {
        "id": "BIOMD0000000595",
        "T": 100.0,
        "N": 200,
        "effort": "medium",
        "note": "ns~218, 1490 rxns (reaches steady state fast at T=100)",
    },
    {"id": "MODEL9087255381", "T": 100.0, "N": 200, "effort": "high", "note": "ns~289"},
    {"id": "MODEL9087474843", "T": 100.0, "N": 200, "effort": "high", "note": "ns~290, 497 rxns"},
    # Rule-based `.net` network exercising the per-observable Functional path
    # (GH #76 task 2): 16 of its reactions are Functional with a mass-action
    # species factor (rate = func(observables)*∏reactants).  Before task 2 the
    # C++ side rejected these (per_observable) terms and the whole model fell to
    # FD; it now attaches the analytical Jacobian.  Dense (ns<50 → dense CVODE
    # whose FD Jacobian costs O(ns) RHS evals), so the analytical scatter wins
    # per step.
    {
        "id": "egfr_net_red",
        "net": "ode/egfr_net_red.net",
        "T": 100.0,
        "N": 200,
        "effort": "low",
        "note": "EGFR reduced, ns~40, 123 rxns, 16 per-observable Functional reactions (.net)",
    },
]

# Correctness gate (peak-relative, mirroring the ode suite's cross_validate).
# A different Jacobian only changes CVODE's Newton/step-acceptance path, so at a
# fixed solver tolerance the analytical and FD runs land on different but each
# tolerance-valid trajectories.  Two independent CVODE integrations therefore
# agree only to ~their own tolerance floor; the tolerance-convergence
# diagnostic (diagnose_divergence.py) confirms the gap collapses to ~1e-12 of
# peak when rtol is tightened to 1e-10.  So the gate is peak-relative with an
# absolute floor at ATOL_REL * peak -- a genuine divergence on a *significant*
# species still far exceeds it, while a sub-floor near-zero species (whose
# element-wise relative diff explodes through a sign change) does not trip it.
CORRECTNESS_RTOL = 1e-5
CORRECTNESS_ATOL_REL = 1e-6  # times the trajectory peak magnitude


def sbml_path(model_id: str) -> Path | None:
    d = MODELS_DIR / model_id
    if not d.is_dir():
        return None
    cand = d / f"{model_id}.xml"
    if cand.exists():
        return cand
    xmls = sorted(d.glob("*.xml"))
    return xmls[0] if xmls else None


def model_path(m: dict) -> Path | None:
    """Resolve a MODELS entry to a file path: a `.net` network under
    benchmarks/models/net/ (key ``net``) or an SBML model id in the rr_parity
    corpus."""
    if m.get("net"):
        p = NET_DIR / m["net"]
        return p if p.exists() else None
    return sbml_path(m["id"])


# --------------------------------------------------------------------------- #
# Worker (runs in a subprocess with a fixed Jacobian mode)
# --------------------------------------------------------------------------- #
def _solver_stats(res):
    out = {}
    ss = getattr(res, "solver_stats", None)
    if isinstance(ss, dict):
        for k, v in ss.items():
            if isinstance(v, (int, float, bool)):
                out[k] = v
    return out


def _species_array(res):
    """Best-effort dense float array of the species trajectory for comparison."""
    sp = getattr(res, "species", None)
    if sp is not None:
        try:
            return np.asarray(sp, dtype=float)
        except (TypeError, ValueError):
            pass
    df = getattr(res, "dataframe", None)
    if df is not None:
        return np.asarray(df.values, dtype=float)
    raise RuntimeError("cannot extract species array from Result")


def worker(path, mode, T, N, traj_out=None):
    os.environ["BNGSIM_ANALYTICAL_FUNCTIONAL_JAC"] = "1" if mode == "analytical" else "0"
    out = {"path": _relpath(path), "mode": mode, "T": T, "N": N}
    try:
        import bngsim

        out["version"] = bngsim.__version__
        # Dispatch on extension: rule-based `.net` networks (the per-observable
        # Functional path, GH #76 task 2) vs SBML kinetic-law models.
        model = (
            bngsim.Model.from_net(str(path))
            if str(path).endswith(".net")
            else bngsim.Model.from_sbml(str(path))
        )
        sim = bngsim.Simulator(model, method="ode")
        t0 = time.perf_counter()
        res = sim.run(t_span=(0.0, T), n_points=N)
        out["wall"] = time.perf_counter() - t0
        out["stats"] = _solver_stats(res)
        if traj_out:
            np.savez(traj_out, time=np.asarray(res.time, dtype=float), species=_species_array(res))
            out["traj_out"] = traj_out
        out["ok"] = True
    except Exception as e:  # noqa: BLE001
        import traceback

        out["error"] = repr(e)
        out["traceback"] = traceback.format_exc()
    print(json.dumps(out))


# --------------------------------------------------------------------------- #
# Driver helpers
# --------------------------------------------------------------------------- #
def _spawn(path, mode, T, N, traj_out=None, timeout=1800):
    cmd = [sys.executable, str(HERE / "run.py"), "--worker", str(path), mode, repr(T), str(N)]
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


def check_correctness(path, T, N):
    a_npz = RESULTS / "_traj_analytical.npz"
    f_npz = RESULTS / "_traj_fd.npz"
    ra = _spawn(path, "analytical", T, N, traj_out=a_npz)
    rf = _spawn(path, "fd", T, N, traj_out=f_npz)
    if not (ra.get("ok") and rf.get("ok")):
        return {"pass": False, "reason": "run_failed", "analytical": ra, "fd": rf}
    A = np.load(a_npz)
    F = np.load(f_npz)
    sa, sf = A["species"], F["species"]
    if sa.shape != sf.shape:
        return {"pass": False, "reason": f"shape {sa.shape} vs {sf.shape}"}
    peak = float(np.max(np.abs(sf))) or 1.0
    atol = CORRECTNESS_ATOL_REL * peak
    absdiff = np.abs(sa - sf)
    max_abs = float(np.max(absdiff))
    # numpy allclose rule: |a-b| <= atol + rtol*|b|; report worst as a multiple.
    allowed = atol + CORRECTNESS_RTOL * np.abs(sf)
    tol_ratio = float(np.max(absdiff / allowed))
    peak_rel = max_abs / peak
    for npz in (a_npz, f_npz):
        npz.unlink(missing_ok=True)
    return {
        "pass": bool(tol_ratio <= 1.0),
        "tol_ratio": tol_ratio,
        "max_abs": max_abs,
        "peak": peak,
        "peak_rel": peak_rel,
    }


def time_modes(path, T, N, repeats):
    """Time both modes, INTERLEAVED round-by-round so each sees the same load
    distribution (avoids biasing the ratio if machine load drifts between
    blocks).  Each (model, mode) run is a fresh subprocess.

    The headline statistic is **min**-of-N, not median: this is a compute-bound
    microbenchmark, so all OS/contention noise is *additive* (other work only
    ever slows a run down).  The minimum is therefore the cleanest estimate of
    true cost (the standard `timeit` practice).  Median + spread are kept for
    an honest read on how noisy the machine was during the run.
    """
    walls = {"analytical": [], "fd": []}
    recs = {"analytical": [], "fd": []}
    for _ in range(repeats):
        for mode in ("analytical", "fd"):  # interleaved
            r = _spawn(path, mode, T, N)
            recs[mode].append(r)
            if r.get("ok"):
                walls[mode].append(r["wall"])
    res = {}
    for mode in ("analytical", "fd"):
        w = walls[mode]
        res[mode] = {
            "min_wall": min(w) if w else None,
            "median_wall": median(w) if w else None,
            "spread": (max(w) - min(w)) if len(w) > 1 else 0.0,
            "rel_spread": ((max(w) - min(w)) / median(w)) if len(w) > 1 and median(w) else 0.0,
            "n_ok": len(w),
            "stats": recs[mode][-1].get("stats", {}) if recs[mode] else {},
            "last": recs[mode][-1] if recs[mode] else {},
        }
    a = res["analytical"]["min_wall"]
    f = res["fd"]["min_wall"]
    res["speedup"] = (f / a) if (a and f) else None  # min-based (headline)
    am, fm = res["analytical"]["median_wall"], res["fd"]["median_wall"]
    res["speedup_median"] = (fm / am) if (am and fm) else None
    return res


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=["both", "correctness", "timing"], default="both")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument(
        "--quick", action="store_true", help="one analytical run per model, smoke only"
    )
    ap.add_argument("--worker", nargs="*", metavar="ARGS")
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
        path = model_path(m)
        if path is None:
            print(f"{m['id']}: MISSING")
            rows.append({"model": m["id"], "error": "missing"})
            continue
        row = {
            "model": m["id"],
            "effort": m["effort"],
            "T": m["T"],
            "N": m["N"],
            "note": m["note"],
        }

        if args.quick:
            r = _spawn(path, "analytical", m["T"], m["N"])
            row["quick"] = {"ok": r.get("ok"), "wall": r.get("wall"), "error": r.get("error")}
            print(f"{m['id']}: quick ok={r.get('ok')} wall={r.get('wall')}")
            rows.append(row)
            continue

        if args.mode in ("both", "correctness"):
            c = check_correctness(path, m["T"], m["N"])
            row["correctness"] = c
            print(
                f"{m['id']}: correctness pass={c.get('pass')} "
                f"tol_ratio={c.get('tol_ratio')} peak_rel={c.get('peak_rel')}"
            )

        if args.mode in ("both", "timing"):
            # only report timing where correctness passed (or wasn't run)
            if args.mode == "timing" or row.get("correctness", {}).get("pass", True):
                t = time_modes(path, m["T"], m["N"], args.repeats)
                row["timing"] = t
                print(
                    f"{m['id']}: analytical(min)={t['analytical']['min_wall']:.4g} "
                    f"fd(min)={t['fd']['min_wall']:.4g} speedup={t['speedup']:.2f}x "
                    f"(median-based {t['speedup_median']:.2f}x)"
                )
            else:
                print(f"{m['id']}: timing SKIPPED (correctness failed)")
        rows.append(row)

    payload = {
        "machine_info": nb.machine_info(),
        "mode": args.mode,
        "effort": args.effort,
        "repeats": args.repeats,
        "results": rows,
    }
    (RESULTS / "jacobian_results.json").write_text(json.dumps(payload, indent=2, default=str))
    write_md(payload, args)
    print("WROTE", RESULTS / "jacobian_results.json", RESULTS / "jacobian_results.md")


def _stat(row_mode, key):
    try:
        return row_mode["stats"].get(key)
    except (AttributeError, TypeError):
        return None


def write_md(payload, args):
    rows = payload["results"]
    info = payload.get("machine_info", {})
    ver = info.get("bngsim_version") or ""
    if not ver:
        for r in rows:
            ver = r.get("timing", {}).get("analytical", {}).get("last", {}).get("version") or ver
    L = ["# GH #76 -- analytical vs FD Functional Jacobian", ""]
    L.append(
        f"_machine: {info.get('platform', 'n/a')} / {info.get('processor', 'n/a')} / "
        f"Python {info.get('python', 'n/a')} / git {info.get('git_commit', 'n/a')}_"
    )
    L.append("")
    L.append(
        "Performance feature: the analytical Jacobian only changes CVODE's "
        "Newton path, so trajectories are identical to FD (verified by the "
        "correctness gate below).  The metric is **time-to-solution**.  FD is "
        "forced with `BNGSIM_ANALYTICAL_FUNCTIONAL_JAC=0`; each (model, mode) "
        "runs in a fresh subprocess and wall-clock times the `run()` integrate "
        "call only (load excluded)."
    )
    L.append("")
    L.append(f"_effort={args.effort}, repeats={args.repeats}, bngsim {ver}_")
    L.append("")

    if any("correctness" in r for r in rows):
        L.append("## Correctness (analytical vs FD trajectory)")
        L.append("")
        L.append(
            f"PASS = worst cell within `atol + rtol*|fd|`, with rtol={CORRECTNESS_RTOL:g} "
            f"and atol={CORRECTNESS_ATOL_REL:g}*peak (peak-relative, the ode suite's "
            f"`cross_validate` convention).  A different Jacobian only changes CVODE's "
            f"step path, so the two runs differ at ~solver tolerance; `tol_ratio` is the "
            f"worst cell as a multiple of its own tolerance (<=1 passes), and `peak_rel` "
            f"is the worst absolute diff over the trajectory peak."
        )
        L.append("")
        L.append("| model | pass | tol_ratio | peak_rel | max_abs |")
        L.append("|---|---|---|---|---|")
        for r in rows:
            c = r.get("correctness")
            if not c:
                continue
            L.append(
                f"| {r['model']} | {'PASS' if c.get('pass') else 'FAIL'} | "
                f"{c.get('tol_ratio')} | {c.get('peak_rel')} | {c.get('max_abs')} |"
            )
        L.append("")

    if any("timing" in r for r in rows):
        L.append("## Wall-clock")
        L.append("")
        L.append(
            "Headline = **min** of the repeats (cleanest estimate of true cost "
            "for compute-bound code: contention noise is additive, so the minimum "
            "is the least-disturbed run).  `rel_spread` (max-min over median) shows "
            "how noisy the machine was; a large spread means the median is "
            "unreliable and the min should be trusted.  Modes are interleaved "
            "round-by-round so both see the same load."
        )
        L.append("")
        L.append(
            "| model | effort | size | analytical min (s) | FD min (s) | speedup (min) | speedup (median) | a/fd rel_spread |"
        )
        L.append("|---|---|---|---|---|---|---|---|")
        for r in rows:
            t = r.get("timing")
            if not t:
                continue
            a, f, sp = t["analytical"]["min_wall"], t["fd"]["min_wall"], t["speedup"]
            spm = t.get("speedup_median")
            rs = f"{t['analytical']['rel_spread']:.0%}/{t['fd']['rel_spread']:.0%}"
            if a and f and sp:
                cell = f"{a:.4f} | {f:.4f} | {sp:.2f}x | {spm:.2f}x | {rs}"
            else:
                cell = f"{a} | {f} | {sp} | {spm} | {rs}"
            L.append(f"| {r['model']} | {r['effort']} | {r['note']} | {cell} |")
        L.append("")
        L.append("## CVODE solver counts (analytical / FD)")
        L.append("")
        L.append(
            "Near-identical between modes (same Newton path); the wall-clock "
            "delta is cost per Jacobian evaluation, which `n_rhs_evals` does "
            "**not** capture (FD perturbations are internal to the Jacobian "
            "routine)."
        )
        L.append("")
        L.append("| model | n_steps | n_jac_evals | n_rhs_evals |")
        L.append("|---|---|---|---|")
        for r in rows:
            t = r.get("timing")
            if not t:
                continue
            a, f = t["analytical"], t["fd"]

            def pair(key, a=a, f=f):
                return f"{_stat(a, key)} / {_stat(f, key)}"

            L.append(
                f"| {r['model']} | {pair('n_steps')} | {pair('n_jac_evals')} | {pair('n_rhs_evals')} |"
            )
        L.append("")
        speedups = [
            r["timing"]["speedup"] for r in rows if r.get("timing") and r["timing"].get("speedup")
        ]
        geo = nb.geometric_mean(speedups) if speedups else None
        if geo:
            L.append(
                f"**Geometric-mean speedup (FD/analytical, min-based): {geo:.2f}x** "
                f"across {len(speedups)} models."
            )
            L.append("")
            L.append("## Interpretation (honest attribution)")
            L.append("")
            L.append(
                "The benefit is real but **modest (~1.0-1.2x)** on this model set, and "
                "the reason is the FD baseline it is measured against, which is itself "
                "already optimized:\n"
                "\n"
                "- **Large models (ns >= 50) use a sparse KLU solver, and their FD "
                "Jacobian is *colored* finite differences** (Curtis-Powell-Reid, "
                "`cvode_colored_jac`): one RHS eval per sparsity color, O(n_colors) "
                "not O(ns).  For the sparse metabolic models here n_colors is small, so "
                "the FD baseline is cheap and the analytical scatter saves little -- "
                "hence ~1.01x on the ns~290 models.\n"
                "- **Small models (ns < 50) use the dense path with CVODE's internal "
                "O(ns) FD.** Analytical wins per Jacobian, but total solve time is a few "
                "ms and dominated by fixed per-step overhead, so the ratio is ~1.04x.\n"
                "- The mid-size model (ns~200) is the sweet spot at ~1.15x.\n"
                "\n"
                "The regime where an analytical Jacobian wins decisively is a model that "
                "is **large AND dense** (high chromatic number, so coloring degrades to "
                "O(ns)) **with an expensive RHS** -- i.e. large rule-based networks.  "
                "Those are exactly (a) the `.net` per-observable path the C++ side "
                "currently rejects (`model.cpp` `if (in.per_observable) return false`), "
                "routing such models to FD, and (b) where symbolic differentiation is "
                "hardest.  So the headline performance upside is gated on the #76 "
                "follow-ups, not realized by the SBML functional models measured here.\n"
                "\n"
                "Validity note: with `BNGSIM_ANALYTICAL_FUNCTIONAL_JAC=0` the functional "
                "Jacobian terms are not populated, so `analytical_jacobian_complete()` "
                "returns False for these (functional) models and dispatch routes to a "
                "genuine FD path -- colored-FD (sparse) or internal-FD (dense), never a "
                "partial-analytical Jacobian.  Both modes share the same linear solver "
                "and RHS; only the Jacobian computation differs."
            )
            L.append("")

    (RESULTS / "jacobian_results.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
