#!/usr/bin/env python3
"""Phase 2 of the full-network ODE timing benchmark: time BNGsim warm integration
vs. BioNetGen ``run_network`` on the FULL networks cached by Phase 1.

Reads ``nets_manifest.json`` + the ``nets/*.net`` cache (Phase 1). For every model
that generated a network, it does a UNIFORM single-segment warm ODE solve — this is
deliberately simpler and more comparable than the parity report, which mixes
single-segment warm solves with coarse multi-phase replays. Both engines integrate
the SAME cached ``.net`` over the SAME horizon, so the per-model cost ratio is fair.

  * horizon: the model's own representative ODE simulate (``parse_ode_spec``);
    models that shipped without a runnable ODE protocol (parity REHAB entries) use
    a documented default (``--default-tend`` / ``--default-nsteps``), flagged in the
    output as ``horizon_source="default"``.
  * BNGsim: ``_bng_common.bn_ode_net`` — cold + warm reps; we report the warm median
    and the linear solver BNGsim auto-selected (Dense vs KLU).
  * run_network: ``_bng_common.run_network_ode`` — the legacy CVODE binary on the
    same ``.net``.

Output ``report_ode_timing_fullnet.json`` matches the schema the manuscript figures
consume (``timing.bngsim`` / ``timing.run_network`` / ``timing.spec``; ``outcome``
== ``PASS`` when both engines produced a finite trajectory). RESUMABLE: a model
already timed (PASS) is skipped unless ``--redo``.

    export BNGPATH=~/Simulations/BioNetGen-2.9.3
    ~/Code/bngsim/.venv/bin/python run_timing.py --workers 4
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import threading
import time
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
OUT = HERE / "report_ode_timing_fullnet.json"
JOBS = BNG_PARITY / "jobs.json"
MODELS_ROOT = BNG_PARITY / "models"

_lock = threading.Lock()


def resolve_run_network() -> str:
    if os.environ.get("RUN_NETWORK"):
        p = Path(os.environ["RUN_NETWORK"])
    else:
        root = Path(
            os.environ.get("BNGPATH", str(Path.home() / "Simulations" / "BioNetGen-2.9.3"))
        )
        root = root.parent if root.name == "BNG2.pl" else root
        p = root / "bin" / "run_network"
    if not p.exists():
        sys.exit(f"ABORT: run_network not found at {p} (set BNGPATH to the BioNetGen-2.9.3 root)")
    return str(p)


def time_one(model_id: str, net_file: str, model_rel: str, rn_bin: str, args) -> dict:
    """Time BNGsim warm + run_network on one cached full network."""
    res: dict = {"model_id": model_id, "method": "ode", "reference_engine": "bng"}
    net_path = NETS / net_file
    net_params = bc.read_net_parameters(net_path)
    bngl_text = (MODELS_ROOT / model_rel).read_text(errors="replace")

    # Horizon: the model's own representative ODE simulate, else a documented default.
    ode = bc.parse_ode_spec(bngl_text, net_params, atol=args.atol, rtol=args.rtol)
    if ode is None:
        ode = {
            "t_start": 0.0,
            "t_end": args.default_tend,
            "n_steps": args.default_nsteps,
            "atol": args.atol,
            "rtol": args.rtol,
        }
        horizon_source = "default"
    else:
        horizon_source = "model"
    n_points = int(ode["n_steps"]) + 1

    bn = bn_timing = None
    bn_exc = ""
    rn = rn_timing = None
    rn_exc = ""

    workdir = Path(tempfile.mkdtemp(prefix="fullnet_tm_"))
    try:
        try:
            t, v, n, bn_timing = bc.bn_ode_net(
                net_path, ode["t_start"], ode["t_end"], n_points, ode["rtol"], ode["atol"]
            )
            bn = (t, v, n)
        except Exception as exc:  # noqa: BLE001
            bn_exc = f"bngsim: {type(exc).__name__}: {exc}"[:400]

        try:
            t, v, n, rn_timing = bc.run_network_ode(
                net_path,
                rn_bin,
                t_start=ode["t_start"],
                t_end=ode["t_end"],
                n_steps=int(ode["n_steps"]),
                rtol=ode["rtol"],
                atol=ode["atol"],
                out_prefix=str(workdir / "rn"),
                timeout=args.timeout,
            )
            rn = (t, v, n)
        except Exception as exc:  # noqa: BLE001
            rn_exc = f"run_network: {type(exc).__name__}: {exc}"[:400]
    finally:
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)

    n_species = int(bn[1].shape[1]) if bn else (int(rn[1].shape[1]) if rn else None)
    spec = {
        "t_start": ode["t_start"],
        "t_end": ode["t_end"],
        "n_steps": int(ode["n_steps"]),
        "rtol": ode["rtol"],
        "atol": ode["atol"],
        "n_species": n_species,
        "mode": "single_segment",
        "horizon_source": horizon_source,
    }
    timing: dict = {"spec": spec}
    if bn_timing:
        timing["bngsim"] = bn_timing
    if rn_timing:
        timing["run_network"] = rn_timing

    bn_ok = bn is not None and np.isfinite(bn[1]).all()
    rn_ok = rn is not None and np.isfinite(rn[1]).all()
    if bn_ok and rn_ok:
        outcome = "PASS"
    elif not bn_ok and rn_ok:
        outcome = "BNGSIM_FAIL"
    elif bn_ok and not rn_ok:
        outcome = "RN_FAIL"
    else:
        outcome = "BOTH_FAIL"
    res.update({"outcome": outcome, "timing": timing})
    if bn_exc or rn_exc:
        res["exception"] = "; ".join(x for x in (bn_exc, rn_exc) if x)[:600]
    return res


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--timeout", type=float, default=300.0, help="Per-engine subprocess timeout (s)."
    )
    ap.add_argument("--rtol", type=float, default=bc.DEFAULT_RTOL)
    ap.add_argument("--atol", type=float, default=bc.DEFAULT_ATOL)
    ap.add_argument(
        "--default-tend",
        type=float,
        default=100.0,
        help="t_end for models with no author ODE horizon (REHAB'd).",
    )
    ap.add_argument(
        "--default-nsteps",
        type=int,
        default=100,
        help="n_steps for models with no author ODE horizon (REHAB'd).",
    )
    ap.add_argument("--models", default="", help="Comma-separated model_id substring filter.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--redo", action="store_true", help="Re-time models already PASS.")
    args = ap.parse_args()

    rn_bin = resolve_run_network()
    if not NETS_MANIFEST.exists():
        sys.exit(f"missing {NETS_MANIFEST}; run gen_networks.py (Phase 1) first.")
    nets = json.loads(NETS_MANIFEST.read_text())
    _meta, alljobs = read_manifest(JOBS)
    model_rel = {j.model_id: j.model for j in alljobs}

    prior = {}
    if OUT.exists():
        prior = {r["model_id"]: r for r in json.loads(OUT.read_text()).get("results", [])}

    todo = []
    for mid, row in nets.items():
        if row.get("status") != "ok" or not row.get("net_file"):
            continue
        if args.models and not any(s.strip() in mid for s in args.models.split(",")):
            continue
        if not args.redo and prior.get(mid, {}).get("outcome") == "PASS":
            continue
        todo.append((mid, row["net_file"], model_rel.get(mid, mid)))
    if args.limit:
        todo = todo[: args.limit]

    print("=" * 72)
    print("  Phase 2 — full-network ODE timing (BNGsim warm vs run_network)")
    print("=" * 72)
    print(
        f"  cached networks: {sum(1 for r in nets.values() if r.get('status') == 'ok')}   to time: {len(todo)}"
    )
    print(f"  run_network: {rn_bin}   workers: {args.workers}")
    print()
    if not todo:
        print("nothing to do.")
        return 0

    results = dict(prior)
    done = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(time_one, mid, nf, mr, rn_bin, args): mid for mid, nf, mr in todo}
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            with _lock:
                results[r["model_id"]] = r
                _write(results, args)
            bn = r["timing"].get("bngsim") or {}
            warm = bn.get("integrate_warm_median_sec")
            ls = (bn.get("config") or {}).get("linear_solver")
            rn = (r["timing"].get("run_network") or {}).get("integrate_sec")
            nsp = (r["timing"].get("spec") or {}).get("n_species")
            print(
                f"  [{done}/{len(todo)}] {r['outcome']:11} N={str(nsp):>5} "
                f"warm={_fmt(warm)} rn={_fmt(rn)} {str(ls):6} "
                f"{r['model_id'].split('/')[-1][:40]}"
            )

    dt = time.perf_counter() - t0
    from collections import Counter

    tally = Counter(r["outcome"] for r in results.values())
    print(f"\n  timed {done} in {dt:.0f}s. outcomes: {dict(tally)}\n  report: {OUT}")
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
    from collections import Counter

    doc = {
        "_meta": {
            "suite": "ode_fullnet",
            "description": "Full-network ODE timing: BNGsim warm vs run_network, uniform single-segment warm solve.",
            "reference_label": "run_network",
            "versions": ver,
            "n_results": len(ordered),
            "tally": dict(Counter(r["outcome"] for r in ordered)),
            "rtol": args.rtol,
            "atol": args.atol,
            "default_horizon": {"t_end": args.default_tend, "n_steps": args.default_nsteps},
        },
        "results": ordered,
    }
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(OUT)


if __name__ == "__main__":
    raise SystemExit(main())
