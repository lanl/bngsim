#!/usr/bin/env python3
"""Phase 3 of the full-network ODE timing benchmark: warm BNGsim integration cost
with the linear solver FORCED, rather than auto-selected.

Phase 2 (``run_timing.py``) measures BNGsim under its shipping policy, which picks
sparse KLU only when a network is both large (``N >= 50``) and sparse (density
``< 0.10``) and dense otherwise. That single arm cannot separate two different
claims: that the sparse solver is *worth having* on large sparse networks, and that
the *selection rule* picks correctly. Answering both needs the same corpus re-timed
with each solver forced on every model.

Runs BNGsim only — ``run_network`` is dense-only, already measured in Phase 2, and
its process-spawn cost would just add noise. Everything else (cached ``.net``,
horizon, tolerances, adaptive warm-rep policy) is held identical to Phase 2, so a
per-model point is comparable across panels.

  --mode dense   force the dense direct solver (``force_dense_linear_solver``)
  --mode sparse  force sparse KLU on every model (``force_sparse_linear_solver``,
                 lanl/bngsim#29). Against an installed bngsim predating that
                 flag this mode aborts with a clear message rather than silently
                 recording auto-selected numbers.
  --mode auto    re-measure the shipping policy through this same script (a
                 cross-check on Phase 2, not a published arm)

Horizons come from the Phase 2 report when it exists, so every panel integrates the
identical span; a model absent there falls back to re-deriving from its BNGL.

Output ``report_ode_timing_forced_<mode>.json`` matches the schema the manuscript
figure consumes. Unlike Phase 2, a model that fails or times out is RECORDED (with
``outcome`` ``FAIL``/``TIMEOUT``) instead of dropped — a forced configuration being
unusable on a model is a result, and the figure marks it. RESUMABLE: a model with a
terminal outcome is skipped unless ``--redo``.

    export BNGPATH=~/Simulations/BioNetGen-2.9.3
    ~/Code/bngsim/.venv/bin/python run_forced.py --mode dense --workers 1
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

BNGSIM = Path(os.environ.get("BNGSIM_ROOT", Path.home() / "Code" / "bngsim"))
PARITY = BNGSIM / "parity_checks"
BNG_PARITY = PARITY / "bng_parity"
sys.path.insert(0, str(PARITY))
sys.path.insert(0, str(BNG_PARITY))
import _bng_common as bc  # noqa: E402
from _core import read_manifest, versions  # noqa: E402

HERE = Path(__file__).resolve().parent
NETS = HERE / "nets"
NETS_MANIFEST = HERE / "nets_manifest.json"
AUTO_REPORT = HERE / "report_ode_timing_fullnet.json"
JOBS = BNG_PARITY / "jobs.json"
MODELS_ROOT = BNG_PARITY / "models"

_lock = threading.Lock()

# Simulator kwarg that pins each mode. "auto" passes nothing.
MODE_KWARG = {
    "dense": "force_dense_linear_solver",
    "sparse": "force_sparse_linear_solver",
    "auto": None,
}


def out_path(mode: str) -> Path:
    return HERE / f"report_ode_timing_forced_{mode}.json"


def check_mode_supported(mode: str) -> None:
    """Abort unless the installed bngsim can actually pin this mode.

    Guards the failure that would quietly corrupt the figure: asking for a forced
    mode the build does not implement, getting auto-selected numbers back, and
    plotting them as if they were forced.
    """
    import inspect

    import bngsim

    if not bngsim.capabilities()["features"].get("klu"):
        sys.exit(
            "ABORT: this bngsim build has no KLU, so every solve is dense and no "
            "mode is distinguishable. Rebuild with SuiteSparse."
        )
    kwarg = MODE_KWARG[mode]
    if kwarg is None:
        return
    params = inspect.signature(bngsim.Simulator.__init__).parameters
    if kwarg not in params:
        sys.exit(
            f"ABORT: bngsim {bngsim.__version__} has no Simulator(..., {kwarg}=...), so "
            f"--mode {mode} cannot be pinned.\n"
            f"       See lanl/bngsim#29; land it and rebuild the extension first."
        )


def _horizon_from_auto_report() -> dict:
    """model_id -> Phase 2 ``timing.spec``, so the panels share exact horizons."""
    if not AUTO_REPORT.exists():
        return {}
    doc = json.loads(AUTO_REPORT.read_text())
    return {
        r["model_id"]: (r.get("timing") or {}).get("spec") or {}
        for r in doc.get("results", [])
        if (r.get("timing") or {}).get("spec")
    }


def resolve_horizon(model_id: str, net_path: Path, model_rel: str, auto_spec: dict, args) -> dict:
    """(t_start, t_end, n_steps, rtol, atol, horizon_source) for one model."""
    spec = auto_spec.get(model_id)
    if spec and spec.get("t_end") is not None:
        return {
            "t_start": float(spec["t_start"]),
            "t_end": float(spec["t_end"]),
            "n_steps": int(spec["n_steps"]),
            "rtol": float(spec.get("rtol", args.rtol)),
            "atol": float(spec.get("atol", args.atol)),
            "horizon_source": spec.get("horizon_source", "phase2"),
        }
    # No Phase 2 row — re-derive exactly as run_timing.py does.
    net_params = bc.read_net_parameters(net_path)
    bngl_text = (MODELS_ROOT / model_rel).read_text(errors="replace")
    ode = bc.parse_ode_spec(bngl_text, net_params, atol=args.atol, rtol=args.rtol)
    if ode is None:
        return {
            "t_start": 0.0,
            "t_end": args.default_tend,
            "n_steps": args.default_nsteps,
            "rtol": args.rtol,
            "atol": args.atol,
            "horizon_source": "default",
        }
    ode["horizon_source"] = "model"
    return ode


def time_one(model_id: str, net_file: str, model_rel: str, auto_spec: dict, args) -> dict:
    """Cold + warm BNGsim solves on one cached full network with the solver pinned.

    Mirrors ``_bng_common.bn_ode_net`` (same phases, same adaptive warm-rep policy
    via ``_warm_rep_count``, same ``_integrate_stats`` shape) with the one addition
    it does not expose: the forced-solver kwarg.
    """
    import bngsim

    res: dict = {"model_id": model_id, "method": "ode", "mode": args.mode}
    net_path = NETS / net_file
    h = resolve_horizon(model_id, net_path, model_rel, auto_spec, args)
    n_points = int(h["n_steps"]) + 1

    kwargs = {}
    if (kw := MODE_KWARG[args.mode]) is not None:
        kwargs[kw] = True

    timing: dict = {}
    outcome = "PASS"
    exc_txt = ""
    n_species = (auto_spec.get(model_id) or {}).get("n_species")

    try:
        t0 = time.perf_counter()
        model = bngsim.Model.from_net(str(net_path))
        load_sec = time.perf_counter() - t0

        sim = bngsim.Simulator(model, method="ode", **kwargs)

        t1 = time.perf_counter()
        r = sim.run(
            t_span=(h["t_start"], h["t_end"]),
            n_points=n_points,
            rtol=h["rtol"],
            atol=h["atol"],
            timeout=args.timeout,
        )
        cold_sec = time.perf_counter() - t1

        warm: list[float] = []
        for _ in range(bc._warm_rep_count(cold_sec)):
            try:
                model.reset()
                t1 = time.perf_counter()
                sim.run(
                    t_span=(h["t_start"], h["t_end"]),
                    n_points=n_points,
                    rtol=h["rtol"],
                    atol=h["atol"],
                    timeout=args.timeout,
                )
                warm.append(time.perf_counter() - t1)
            except Exception:  # noqa: BLE001 — a warm rep failing just ends the reps
                break

        values = np.asarray(r.species)
        n_species = int(values.shape[1])
        stats = r.solver_stats if hasattr(r, "solver_stats") else {}
        ls_code = (stats or {}).get("linear_solver", 0)
        solver = bc.LINEAR_SOLVER_NAMES.get(ls_code, f"kind_{ls_code}")

        timing["bngsim"] = {
            "load_sec": round(load_sec, 6),
            "jac_derive_sec": round(float(sim.last_jacobian_sec), 6),
            "codegen_sec": round(float(sim.last_codegen_sec), 6),
            **bc._integrate_stats(cold_sec, warm),
            "config": {
                "codegen": sim.codegen_backend,
                "jacobian": sim.jacobian_strategy,
                "linear_solver": solver,
                "cached": sim.codegen_cache_hit,
            },
        }
        if not np.isfinite(values).all():
            outcome = "FAIL"
            exc_txt = "non-finite trajectory"
        # The forced kwarg is advisory in the C++ gate (a JAX Jacobian or an empty
        # sparsity pattern still wins). Record when the request did not take, so
        # the figure never plots an auto point as a forced one.
        elif args.mode != "auto":
            # LAPACK-dense counts as dense: it is the same O(N^3) factorization
            # via a different BLAS path, opt-in through BNGSIM_LAPACK_DENSE.
            accepted = {"sparse": ("KLU",), "dense": ("Dense", "LAPACK-dense")}[args.mode]
            if solver not in accepted:
                outcome = "NOT_FORCED"
    except bngsim.SimulationTimeout as exc:
        outcome = "TIMEOUT"
        exc_txt = f"{type(exc).__name__}: {exc}"[:400]
    except Exception as exc:  # noqa: BLE001
        # Classify on the exception TYPE, never on the message: a TypeError whose
        # text merely contains "timeout" is a harness bug, and reporting it as a
        # model that timed out would bury it.
        outcome = "FAIL"
        exc_txt = f"{type(exc).__name__}: {exc}"[:400]

    timing["spec"] = {
        "t_start": h["t_start"],
        "t_end": h["t_end"],
        "n_steps": int(h["n_steps"]),
        "rtol": h["rtol"],
        "atol": h["atol"],
        "n_species": n_species,
        "mode": "single_segment",
        "horizon_source": h["horizon_source"],
    }
    res.update({"outcome": outcome, "timing": timing})
    if exc_txt:
        res["exception"] = exc_txt
    return res


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--mode", choices=sorted(MODE_KWARG), required=True)
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Keep at 1 for published numbers; concurrency perturbs warm timings.",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-solve wall limit (s). A forced-dense solve on the largest networks "
        "is the one that can hit this; exceeding it is recorded, not fatal.",
    )
    ap.add_argument("--rtol", type=float, default=bc.DEFAULT_RTOL)
    ap.add_argument("--atol", type=float, default=bc.DEFAULT_ATOL)
    ap.add_argument("--default-tend", type=float, default=100.0)
    ap.add_argument("--default-nsteps", type=int, default=100)
    ap.add_argument("--models", default="", help="Comma-separated model_id substring filter.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    check_mode_supported(args.mode)

    if not NETS_MANIFEST.exists():
        sys.exit(f"missing {NETS_MANIFEST}; run gen_networks.py (Phase 1) first.")
    nets = json.loads(NETS_MANIFEST.read_text())
    _meta, alljobs = read_manifest(JOBS)
    model_rel = {j.model_id: j.model for j in alljobs}
    auto_spec = _horizon_from_auto_report()

    out = out_path(args.mode)
    prior = {}
    if out.exists():
        prior = {r["model_id"]: r for r in json.loads(out.read_text()).get("results", [])}

    todo = []
    for mid, row in nets.items():
        if row.get("status") != "ok" or not row.get("net_file"):
            continue
        if args.models and not any(s.strip() in mid for s in args.models.split(",")):
            continue
        if not args.redo and mid in prior:
            continue
        todo.append((mid, row["net_file"], model_rel.get(mid, mid)))
    # Cheapest first, so a long forced-dense tail does not block early feedback.
    todo.sort(key=lambda t: (auto_spec.get(t[0]) or {}).get("n_species") or 0)
    if args.limit:
        todo = todo[: args.limit]

    print("=" * 72)
    print(f"  Phase 3 — forced-solver ODE timing (BNGsim, --mode {args.mode})")
    print("=" * 72)
    print(
        f"  cached networks: {sum(1 for r in nets.values() if r.get('status') == 'ok')}"
        f"   to time: {len(todo)}   workers: {args.workers}"
    )
    print()
    if not todo:
        print("nothing to do.")
        return 0

    results = dict(prior)
    done = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(time_one, mid, nf, mr, auto_spec, args): mid for mid, nf, mr in todo}
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            with _lock:
                results[r["model_id"]] = r
                _write(results, args)
            bn = r["timing"].get("bngsim") or {}
            warm = bn.get("integrate_warm_median_sec")
            ls = (bn.get("config") or {}).get("linear_solver")
            nsp = (r["timing"].get("spec") or {}).get("n_species")
            print(
                f"  [{done}/{len(todo)}] {r['outcome']:10} N={str(nsp):>5} "
                f"warm={_fmt(warm)} cold={_fmt(bn.get('integrate_cold_sec'))} {str(ls):6} "
                f"{r['model_id'].split('/')[-1][:38]}",
                flush=True,
            )

    dt = time.perf_counter() - t0
    tally = Counter(r["outcome"] for r in results.values())
    print(f"\n  timed {done} in {dt:.0f}s. outcomes: {dict(tally)}\n  report: {out}")
    return 0


def _fmt(x):
    return f"{x * 1000:8.3f}ms" if isinstance(x, (int, float)) and x else "     --  "


def _write(results: dict, args) -> None:
    ver = versions.stamp("bng")
    with contextlib.suppress(Exception):
        ver["sundials"] = bc.sundials_version()
    ordered = sorted(
        results.values(), key=lambda r: ((r["timing"].get("spec") or {}).get("n_species") or 0)
    )
    doc = {
        "_meta": {
            "suite": "ode_fullnet",
            "phase": "forced_solver",
            "mode": args.mode,
            "description": (
                f"Full-network ODE timing, BNGsim only, linear solver forced to "
                f"'{args.mode}'. Same cached .net, horizons, tolerances and warm-rep "
                f"policy as the Phase 2 auto report."
            ),
            "versions": ver,
            "n_results": len(ordered),
            "tally": dict(Counter(r["outcome"] for r in ordered)),
            "rtol": args.rtol,
            "atol": args.atol,
            "per_solve_timeout_sec": args.timeout,
            "workers": args.workers,
            "default_horizon": {"t_end": args.default_tend, "n_steps": args.default_nsteps},
        },
        "results": ordered,
    }
    out = out_path(args.mode)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(out)


if __name__ == "__main__":
    raise SystemExit(main())
