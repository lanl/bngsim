#!/usr/bin/env python3
"""Emit ``_core`` GOLDEN references for the rr_parity suite (bngsim-only).

Golden references are the per-job bngsim reference data that downstream
consumers (PyBNF, PyBioNetGen) regenerate through their own bridges to verify
they reproduce bngsim — *without* needing a RoadRunner install for comparison.
This is pure bngsim output: no RoadRunner side, no differ. (It is the report,
``rr_run.py``, that runs RoadRunner.)

Contract (mirrors bng_parity's ``parity_golden.py``)
----------------------------------------------------
* **One trajectory per job.** An rr_parity job is one SBML × one method, so it
  produces exactly one ``(time, values, names)`` trajectory — there is no
  multi-artifact (.gdat/.scan/.cdat) fan-out like the BNGL suite. The job
  ``checksum`` is a byte-identity hash of that trajectory rounded to
  ``CHECKSUM_SIGFIGS``; the ``fingerprint`` is the per-variable numeric summary
  (``_core.fingerprint.fingerprint``) used as the cross-platform tolerance
  fallback (``fingerprint_max_rel``).
* **Stochastic = single fixed ``seed=1`` trajectory**, NOT an ensemble — so the
  checksum is a true byte-identity key within a pinned (version, platform,
  seed) cell. bngsim's C++ re-seeds deterministically, so the same model state
  + ``seed=1`` reproduces a trajectory exactly (GH #10 seed semantics).
* **Full-trajectory subset:** only the hand-picked ``TRAJECTORY_ALLOWLIST``
  (keyed ``model_id:method``, each with a one-line reason) gets its full
  ``(time, names, values)`` written under ``golden/trajectories/``; every other
  job gets checksum + fingerprint only. Allow-listed jobs absent from the run
  are emitted without a trajectory (no error), so the list can be aspirational.

Isolation: each bngsim run is its own spawned subprocess with a wall-clock cap
(shared scheduler in ``_rr_common``), so a pathological model can't wedge the
whole golden pass.

Usage:
    cd bngsim && .venv/bin/python parity_checks/rr_parity/rr_golden.py \\
        --regime ode --models BIOMD0000000001,BIOMD0000000012 --workers 2
    .venv/bin/python parity_checks/rr_parity/rr_golden.py --regime both --workers 4

Output: golden/golden.json + golden/trajectories/<id>__<method>.json (committed).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # so `from _core import ...` resolves
sys.path.insert(0, str(HERE))  # so the spawned child can import _rr_common / rr_golden

import _rr_common as rc  # noqa: E402
from _core import Golden, read_manifest, versions, write_golden  # noqa: E402
from _core.fingerprint import CHECKSUM_SIGFIGS, _round_sig, fingerprint  # noqa: E402
from _core.fingerprint import checksum as _checksum  # noqa: E402
from _core.versions import git_rev  # noqa: E402

ODE_JOBS = HERE / "ode_jobs.json"
SSA_JOBS = HERE / "ssa_jobs.json"
DEFAULT_SEED = 1  # D2: the pinned stochastic trajectory (manifest _meta.defaults.seed)
DEFAULT_ODE_TIMEOUT = 120.0


# Hand-picked representative subset that gets a FULL committed trajectory.
# Keyed "model_id:method" (a model can be both an ODE and an SSA job). Each
# entry: a one-line reason. Entries absent from the consumed run are emitted
# without a trajectory rather than erroring.
TRAJECTORY_ALLOWLIST: dict[str, str] = {
    # --- deterministic (ode) ------------------------------------------------
    "BIOMD0000000001:ode": "Edelstein ACh receptor — the canonical first BioModel; small smoke trajectory",
    "BIOMD0000000010:ode": "Kholodenko MAPK ultrasensitivity cascade — classic stiff multi-scale ODE",
    "BIOMD0000000012:ode": "Elowitz–Leibler repressilator — limit-cycle oscillator, sharp periodic features",
    "BIOMD0000000042:ode": "outputStartTime=120 > initialTime=0 — exercises integration from initial_time (GH #19)",
    # --- stochastic (ssa, seed=1) ------------------------------------------
    "BIOMD0000000010:ssa": "MAPK cascade under SSA at seed=1 — discrete-event trajectory",
    "BIOMD0000000012:ssa": "repressilator under SSA at seed=1 — stochastic oscillation",
}


def _golden_worker(spec: dict, q) -> None:
    """Run bngsim once for one job; put (checksum, fingerprint, [trajectory]) on ``q``.

    Module-level for spawn-pickling. For SSA the run is a single ``seed=1``
    replicate (D2). The trajectory is rounded to ``CHECKSUM_SIGFIGS`` before the
    checksum so the committed reference matches exactly what the checksum hashes.
    """
    rc.set_rr_quiet()
    import bngsim  # noqa: E402  (lazy: confine the C extension to the child)

    p = spec["params"]
    xml = spec["xml"]
    method = spec["method"]
    res = {k: spec[k] for k in ("key", "model_id", "method", "allowlisted")}
    try:
        t0 = time.perf_counter()
        if method == "ode":
            # Integrate from the SED-ML initialTime (GH #19), matching rr_run: the
            # golden must be the trajectory the parity check validates. initial_time
            # == t_start except for outputStartTime > initialTime models; .get keeps
            # pre-field job files integrating from t_start.
            t, values, names = rc.bn_ode(
                xml,
                p.get("initial_time", p["t_start"]),
                p["t_end"],
                p["n_points"],
                p["rtol"],
                p["atol"],
            )
        else:
            # Integrate from the SED-ML initialTime (GH #21, SSA counterpart of
            # #19): the golden must integrate [initial_time, t_end] so any
            # pre-outputStartTime dynamics/events fire. initial_time == t_start
            # except for the outputStartTime > initialTime models; .get keeps
            # pre-field job files integrating from t_start.
            t, reps, names = rc.bn_ssa(
                xml,
                p.get("initial_time", p["t_start"]),
                p["t_end"],
                p["n_points"],
                1,
                DEFAULT_SEED,
            )
            values = reps[0]
        res["wall"] = round(time.perf_counter() - t0, 3)
    except bngsim.SsaValidationError as exc:
        res["status"] = "unsupported"
        res["reason"] = f"SsaValidationError: {exc}"[:300]
        q.put(res)
        return
    except Exception as exc:
        res["status"] = "exception"
        res["reason"] = f"{type(exc).__name__}: {exc}"[:300]
        q.put(res)
        return

    if values.ndim != 2 or values.shape[1] < 1 or values.shape[0] == 0:
        res["status"] = "exception"
        res["reason"] = f"degenerate result shape {values.shape}"
        q.put(res)
        return

    res["status"] = "ok"
    res["checksum"] = _checksum(t, values, names)
    res["fingerprint"] = fingerprint(t, values, names)
    if spec["allowlisted"]:
        # Round to the checksum's sig figs so the committed reference is exactly
        # what the checksum hashed; carried as plain lists across the queue.
        res["traj"] = {
            "time": _round_sig(np.asarray(t, float), CHECKSUM_SIGFIGS).tolist(),
            "names": list(names),
            "values": _round_sig(np.asarray(values, float), CHECKSUM_SIGFIGS).tolist(),
        }
    q.put(res)


def _traj_relpath(key: str) -> str:
    """Suite-relative (under golden/) trajectory path for a 'model_id:method' key."""
    return f"trajectories/{key.replace(':', '__')}.json"


def _downsample(traj: dict, stride: int) -> dict:
    """Keep every ``stride``-th row (the final row always kept)."""
    n = len(traj["time"])
    idx = list(range(0, n, stride))
    if idx and idx[-1] != n - 1:
        idx.append(n - 1)
    return {
        "time": [traj["time"][i] for i in idx],
        "names": traj["names"],
        "values": [traj["values"][i] for i in idx],
    }


def _traj_payload(model_id, method, reason, traj, stride) -> str:
    payload = {
        "model_id": model_id,
        "method": method,
        "seed": DEFAULT_SEED if method == "ssa" else None,
        "reason": reason,
        "sigfigs": CHECKSUM_SIGFIGS,
        **(_downsample(traj, stride) if stride > 1 else traj),
    }
    if stride > 1:
        payload["downsampled_stride"] = stride
    return json.dumps(payload, separators=(",", ":")) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--regime", choices=("ode", "ssa", "both"), default="both")
    ap.add_argument(
        "--golden-dir",
        default=str(HERE / "golden"),
        help="Output dir for golden.json + trajectories/ (committed).",
    )
    ap.add_argument("--workers", type=int, default=4, help="Concurrent bngsim subprocesses.")
    ap.add_argument("--timeout", type=float, default=None, help="Per-job wall cap (s).")
    ap.add_argument(
        "--rtol",
        type=float,
        default=rc.DEFAULT_RTOL,
        help=f"ODE integration rtol (default {rc.DEFAULT_RTOL}; must match rr_run's "
        "forced tol so the golden is computed at the validated accuracy).",
    )
    ap.add_argument(
        "--atol",
        type=float,
        default=rc.DEFAULT_ATOL,
        help=f"ODE integration atol (default {rc.DEFAULT_ATOL}).",
    )
    ap.add_argument(
        "--effort",
        choices=list(rc.EFFORT_LEVELS),
        default="high",
        help="Prune cost-tiered SSA jobs (no-op for ODE).",
    )
    ap.add_argument("--limit", type=int, default=0, help="Max jobs after filtering (0=all).")
    ap.add_argument("--models", default="", help="Comma-separated model_id filter.")
    ap.add_argument("--include", default="", help="Substring filter on the model path.")
    ap.add_argument("--exclude", default="", help="Substring filter — drop matching model paths.")
    ap.add_argument(
        "--max-traj-kb",
        type=int,
        default=950,
        help="Per-trajectory-file size cap (KB); rows are downsampled to fit "
        "(stays under the 1 MB large-file hook). golden.json keeps full resolution.",
    )
    args = ap.parse_args()

    regimes = ("ode", "ssa") if args.regime == "both" else (args.regime,)
    jobs = []
    for regime in regimes:
        path = ODE_JOBS if regime == "ode" else SSA_JOBS
        if not path.exists():
            sys.exit(f"missing {path}; run build_{regime}_jobs.py first.")
        _meta, rjobs = read_manifest(path)
        jobs += rc.load_and_filter(rjobs, args, suite_dir=HERE)
    if not jobs:
        sys.exit("no jobs after filtering.")

    missing = [j.model_id for j in jobs if not rc.model_path(HERE, j).exists()]
    if missing:
        sys.exit(
            f"{len(missing)} job(s) have no vendored SBML (e.g. {missing[:3]}). "
            "Run `python materialize.py` to place the gitignored model tree."
        )

    n_tol_ov = 0
    specs = []
    for j in jobs:
        key = f"{j.model_id}:{j.method}"
        cap = (
            args.timeout
            if args.timeout is not None
            else j.params.get("max_wall_sec") or DEFAULT_ODE_TIMEOUT
        )
        params = dict(j.params)
        # Match rr_run's forced tight tolerance so the golden trajectory is the
        # one the parity check actually validates (SED-ML tol stays provenance).
        # A per-model TOL_OVERRIDES entry (baked into Job.overrides) wins, so the
        # golden is computed at exactly the tol rr_run validates the model at.
        params["rtol"], params["atol"] = args.rtol, args.atol
        for o in j.overrides:
            if o.field == "tol":
                params["rtol"] = float(o.value["rtol"])
                params["atol"] = float(o.value["atol"])
                n_tol_ov += 1
        specs.append(
            {
                "key": key,
                "model_id": j.model_id,
                "method": j.method,
                "xml": str(rc.model_path(HERE, j)),
                "params": params,
                "allowlisted": key in TRAJECTORY_ALLOWLIST,
                "cap": float(cap),
            }
        )

    ver = versions.stamp()  # bngsim + python + platform (no reference engine for golden)
    print("=" * 72)
    print("  rr_parity golden references (bngsim-only)")
    print("=" * 72)
    print(
        f"  regime: {args.regime}   jobs: {len(specs)}   workers: {args.workers}   bngsim {ver['bngsim']}"
    )
    print()

    def _progress(finished, total, res):
        st = res.get("status", "?")
        tag = {
            "ok": "OK ",
            "exception": "ERR",
            "unsupported": "UNSUP",
            "timeout": "SLOW",
            "dead": "DEAD",
        }.get(st, st)
        traj = " +traj" if (st == "ok" and res.get("allowlisted")) else ""
        extra = f" {res.get('reason', '')[:60]}" if st in ("exception", "unsupported") else ""
        print(
            f"  [{finished}/{total}] {tag} {res['model_id']} ({res['method']}){traj}{extra}",
            flush=True,
        )

    t0 = time.perf_counter()
    raw = rc.schedule(
        specs,
        _golden_worker,
        workers=args.workers,
        timeout_of=lambda s: s["cap"],
        on_done=_progress,
    )
    elapsed = time.perf_counter() - t0

    golden_dir = Path(args.golden_dir).resolve()
    traj_dir = golden_dir / "trajectories"
    golden: list[Golden] = []
    skipped: dict[str, str] = {}
    n_traj = 0
    cap_bytes = args.max_traj_kb * 1024

    for r in raw:
        key = r["key"]
        if r.get("status") != "ok":
            skipped[key] = {"timeout": "run_timeout", "dead": "worker_died"}.get(
                r.get("status"), r.get("reason", r.get("status", "?"))
            )
            continue
        traj_rel = None
        if key in TRAJECTORY_ALLOWLIST and "traj" in r:
            traj_dir.mkdir(parents=True, exist_ok=True)
            reason = TRAJECTORY_ALLOWLIST[key]
            stride = 1
            text = _traj_payload(r["model_id"], r["method"], reason, r["traj"], stride)
            while len(text.encode()) > cap_bytes and stride < 64:
                stride *= 2
                text = _traj_payload(r["model_id"], r["method"], reason, r["traj"], stride)
            traj_rel = _traj_relpath(key)
            (golden_dir / traj_rel).write_text(text)
            n_traj += 1
            if stride > 1:
                print(
                    f"  downsampled {key} trajectory (stride {stride}) to fit {args.max_traj_kb}KB cap"
                )
        golden.append(
            Golden(
                model_id=r["model_id"],
                method=r["method"],
                checksum=r["checksum"],
                fingerprint=r["fingerprint"],
                trajectory=traj_rel,
            )
        )

    golden.sort(key=lambda g: (g.model_id, g.method))
    real_skips = {k: v for k, v in skipped.items() if v != "out_dir_missing"}
    meta = {
        "suite": "rr_parity",
        "kind": "golden",
        "regime": args.regime,
        "git_rev": git_rev(str(HERE)),
        "versions": ver,
        "seed": DEFAULT_SEED,
        "checksum_sigfigs": CHECKSUM_SIGFIGS,
        "integration_tol": {"rtol": args.rtol, "atol": args.atol},
        "tol_overridden_jobs": n_tol_ov,
        "n_golden": len(golden),
        "n_trajectories": n_traj,
        "elapsed_sec": round(elapsed, 2),
        "contract": (
            "One trajectory per job (SBML × method). checksum = sha256 of the "
            f"trajectory rounded to {CHECKSUM_SIGFIGS} sig figs (byte-identity within "
            "a pinned version/platform/seed cell); fingerprint = per-variable numeric "
            "summary (fingerprint_max_rel-comparable, the cross-platform fallback). "
            "Stochastic = a single fixed seed=1 trajectory, not an ensemble. Full "
            "trajectories only for the TRAJECTORY_ALLOWLIST subset; consumers "
            "regenerate through their bridge on a pinned bngsim and match the checksum."
        ),
    }
    if real_skips:
        meta["skipped"] = dict(sorted(real_skips.items())[:50])

    golden_dir.mkdir(parents=True, exist_ok=True)
    golden_path = golden_dir / "golden.json"
    write_golden(golden_path, golden, meta=meta)
    print()
    print(f"golden: {golden_path}")
    print(f"  golden records: {len(golden)} / {len(specs)} jobs")
    print(f"  trajectories:   {n_traj} (allow-listed)")
    print(f"  skipped:        {len(real_skips)}")
    print(f"  generated:      {_dt.datetime.now().isoformat(timespec='seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
