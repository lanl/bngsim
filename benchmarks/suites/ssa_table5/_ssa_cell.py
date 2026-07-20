#!/usr/bin/env python3
"""One (engine, model) SSA timing cell — the isolation unit of the ssa_table5 harness.

Invoked as a standalone subprocess by run_ssa_timing.py:

    python _ssa_cell.py <engine> <model_key> <out_json>

so a native crash (COPASI segfault) or a hard timeout (RR/COPASI stuck in a C call)
kills only this cell, not the sweep. Measures:

  COLD = wall of [load/parse + build + first run]          (1 run)
  WARM = median wall over the next N runs, reusing the loaded model
         (reset()/reseed + run), N from the model's warm count, stopped early if the
         cumulative warm wall exceeds WARM_BUDGET_SEC.

Per-run cap PER_RUN_CAP_SEC (brief: 120 s): enforced natively for bngsim (run timeout=)
and run_network (subprocess timeout=); best-effort SIGALRM for RR/COPASI (the
orchestrator's hard CELL_WALL_CAP backstop covers a run that ignores the alarm inside a
C call). events/run is the engine's own Gillespie firing count where it exposes one
(bngsim: solver_stats['n_steps']); RR/COPASI/run_network don't, so their events are
back-filled from the bngsim reference (same model+horizon) at aggregation.

The cold result is flushed to <out_json> BEFORE warm starts, so a hard-kill during warm
still yields a usable cold row.
"""

from __future__ import annotations

import contextlib
import json
import signal
import statistics as st
import sys
import time
import warnings
from pathlib import Path

import _ssa_config as C


class _RunTimeout(Exception):
    pass


class run_guard:
    """Best-effort per-run wall cap via SIGALRM (real time). Raises _RunTimeout if the
    guarded call returns control to Python after the cap. A C call that never yields is
    caught instead by the orchestrator's SIGKILL backstop."""

    def __init__(self, cap: float):
        self.cap = cap

    def __handler(self, *_):
        raise _RunTimeout(f"exceeded {self.cap:.0f}s per-run cap")

    def __enter__(self):
        self._old = signal.signal(signal.SIGALRM, self.__handler)
        signal.setitimer(signal.ITIMER_REAL, self.cap)
        return self

    def __exit__(self, *exc):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self._old)
        return False


def _warm_median(secs: list[float]):
    return round(st.median(secs), 6) if secs else None


def _conversion_label(model_key: str, engine: str) -> str:
    m = C.MODELS[model_key]
    if engine == "bngsim":
        return "native"
    if engine == "run_network":
        return "native (.net)" if m["kind"] == "bngl" else "converted (.net<-SBML)"
    # roadrunner / copasi
    return "native (SBML)" if m["kind"] == "sbml" else "converted (SBML<-.net)"


# --------------------------------------------------------------------------- #
# Engine runners. Each returns (cold_sec, ev_cold, warm_secs, ev_warm, load_sec,
# backend, status, error), flushing the cold row via `flush` before warm.
# --------------------------------------------------------------------------- #
def run_bngsim(kind, path, t_end, n_points, warm_n, seed_base, cap, warm_budget, flush):
    import bngsim

    t0 = time.perf_counter()
    model = bngsim.Model.from_net(path) if kind == "net" else bngsim.Model.from_sbml(path)
    load_sec = time.perf_counter() - t0
    sim = bngsim.Simulator(model, method="ssa")

    def one(seed):
        model.reset()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", bngsim.SsaBoundaryWarning)
            t = time.perf_counter()
            r = sim.run(t_span=(0.0, t_end), n_points=n_points, seed=seed, timeout=cap)
            w = time.perf_counter() - t
        ev = int((r.solver_stats or {}).get("n_steps", 0) or 0)
        backend = (getattr(r, "ssa_diagnostics", None) or {}).get(
            "propensity_backend", "interpreted"
        )
        return w, ev, backend

    try:
        cold, ev_cold, backend = one(seed_base)
    except bngsim.SimulationTimeout:
        return None, None, [], [], load_sec, "", "too_slow", f"cold run exceeded {cap:.0f}s"
    flush(cold, ev_cold, [], [], load_sec, backend, "partial", "")

    warm, ev_w, cum = [], [], 0.0
    for i in range(warm_n):
        if cum > warm_budget:
            break
        try:
            w, ev, _ = one(seed_base + 1 + i)
        except bngsim.SimulationTimeout:
            break
        warm.append(w)
        ev_w.append(ev)
        cum += w
    return cold, ev_cold, warm, ev_w, load_sec, backend, "ok", ""


def run_run_network(kind, path, t_end, n_points, warm_n, seed_base, cap, warm_budget, flush):
    import subprocess
    import tempfile

    n_steps = max(int(n_points) - 1, 1)
    step_size = t_end / n_steps
    tmp = tempfile.mkdtemp(prefix="rn_ssa_")

    def one(seed):
        pfx = str(Path(tmp) / f"r{seed}")
        cmd = [
            C.RUN_NETWORK_BIN,
            "-o",
            pfx,
            "-p",
            "ssa",
            "-h",
            str(int(seed)),
            "--cdat",
            "0",
            "--fdat",
            "0",
            "-g",
            path,
            path,
            repr(step_size),
            str(int(n_steps)),
        ]
        t = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=cap)
        w = time.perf_counter() - t
        if proc.returncode != 0:
            tail = (proc.stderr.strip() or proc.stdout.strip() or "").splitlines()
            raise RuntimeError(f"run_network ssa failed: {(tail[-1] if tail else '')[:200]}")
        if not Path(f"{pfx}.gdat").exists():
            raise RuntimeError("run_network produced no .gdat")
        return w

    load_sec = 0.0  # run_network reloads the .net every call; no separable load phase
    try:
        cold = one(seed_base)
    except subprocess.TimeoutExpired:
        return None, None, [], [], load_sec, "", "too_slow", f"cold run exceeded {cap:.0f}s"
    except Exception as e:  # noqa: BLE001
        return None, None, [], [], load_sec, "", "sim_fail", f"{type(e).__name__}: {e}"[:200]
    flush(cold, None, [], [], load_sec, "run_network (C binary)", "partial", "")

    warm, cum = [], 0.0
    for i in range(warm_n):
        if cum > warm_budget:
            break
        try:
            w = one(seed_base + 1 + i)
        except subprocess.TimeoutExpired:
            break
        warm.append(w)
        cum += w
    return cold, None, warm, [], load_sec, "run_network (C binary, fresh process/call)", "ok", ""


def run_roadrunner(kind, path, t_end, n_points, warm_n, seed_base, cap, warm_budget, flush):
    import roadrunner

    roadrunner.Logger.setLevel(roadrunner.Logger.LOG_CRITICAL)  # quiet the event warnings
    t0 = time.perf_counter()
    try:
        rr = roadrunner.RoadRunner(path)
        rr.integrator = "gillespie"
        rr.integrator.variable_step_size = False
    except Exception as e:  # noqa: BLE001
        return None, None, [], [], 0.0, "", "load_error", f"{type(e).__name__}: {e}"[:200]
    load_sec = time.perf_counter() - t0

    def one(seed):
        rr.reset()
        rr.integrator.seed = int(seed)
        t = time.perf_counter()
        with run_guard(cap):
            rr.simulate(0.0, t_end, int(n_points))
        return time.perf_counter() - t

    try:
        cold = one(seed_base)
    except _RunTimeout as e:
        return None, None, [], [], load_sec, "", "too_slow", str(e)
    except Exception as e:  # noqa: BLE001
        return None, None, [], [], load_sec, "", "sim_fail", f"{type(e).__name__}: {e}"[:200]
    flush(cold, None, [], [], load_sec, "roadrunner gillespie", "partial", "")

    warm, cum = [], 0.0
    for i in range(warm_n):
        if cum > warm_budget:
            break
        try:
            w = one(seed_base + 1 + i)
        except (_RunTimeout, Exception):
            break
        warm.append(w)
        cum += w
    return cold, None, warm, [], load_sec, "roadrunner gillespie", "ok", ""


def run_copasi(kind, path, t_end, n_points, warm_n, seed_base, cap, warm_budget, flush):
    import COPASI

    n_steps = max(int(n_points) - 1, 1)
    t0 = time.perf_counter()
    dm = COPASI.CRootContainer.addDatamodel()
    try:
        if not dm.importSBML(path):
            return None, None, [], [], 0.0, "", "load_error", "COPASI importSBML returned False"
    except Exception as e:  # noqa: BLE001
        return None, None, [], [], 0.0, "", "load_error", f"{type(e).__name__}: {e}"[:200]
    # Molecule-count semantics: this corpus is particle-count only, but a model with no
    # SBML unit definitions (every converted-BNGL SBML, and native BIOMD035/Vilar) imports
    # with COPASI's default quantity unit "mol" -> COPASI multiplies amounts by Avogadro
    # ("particle number too big", stochastic init refused). Force "#" (particle number) so
    # amount 1 == 1 particle, matching the .net/molecule-count encoding the other engines use.
    cm = dm.getModel()
    unit_note = ""
    if cm.getQuantityUnit() == "mol":
        cm.setQuantityUnit("#")
        cm.compileIfNecessary()
        unit_note = " [quantity unit forced mol->#: molecule-count semantics]"
    task = dm.getTask("Time-Course")
    task.setMethodType(COPASI.CTaskEnum.Method_directMethod)
    task.setScheduled(True)
    prob = task.getProblem()
    prob.setDuration(float(t_end))
    prob.setStepNumber(int(n_steps))
    prob.setOutputStartTime(0.0)
    prob.setTimeSeriesRequested(True)
    meth = task.getMethod()
    # Raise the internal-step ceiling: the default caps a single output interval at ~1e9
    # firings, which erk (~1e8+ events) trips as "task process failed" (a limit, not a
    # timeout). Higher limit keeps the SAME exact Direct method — it just doesn't
    # artificially abort a long trajectory. Bounded instead by the 120 s per-run cap.
    _mis = meth.getParameter("Max Internal Steps")
    if _mis is not None:
        _mis.setValue(int(2**31 - 1))
    load_sec = time.perf_counter() - t0

    def _seed(s):
        for nm, val in (("Use Random Seed", True), ("Random Seed", int(s))):
            p = meth.getParameter(nm)
            if p is not None:
                p.setValue(val)

    # NOTE: unlike bngsim (native timeout) and RoadRunner (SIGALRM-interruptible simulate),
    # COPASI's process() is NOT signal-interruptible — an in-process SIGALRM corrupts it into
    # an early False ("task process failed") instead of a clean stop. So the 120 s per-run cap
    # is enforced POST-HOC by wall time here; a genuine hang is still bounded by the
    # orchestrator's hard CELL_WALL_CAP (SIGKILL of the whole cell subprocess).
    def one(seed, first):
        _seed(seed)
        t = time.perf_counter()
        if first and not task.initialize(COPASI.CCopasiTask.OUTPUT_UI):
            raise RuntimeError("COPASI task init failed")
        if not task.process(True):
            raise RuntimeError("COPASI task process failed")
        return time.perf_counter() - t

    try:
        cold = one(seed_base, first=True)
    except Exception as e:  # noqa: BLE001
        return None, None, [], [], load_sec, "", "sim_fail", f"{type(e).__name__}: {e}"[:200]
    if cold > cap:  # over the per-run cap: report the measured wall, flag too_slow, skip warm
        return (
            cold,
            None,
            [],
            [],
            load_sec,
            "COPASI Direct method" + unit_note,
            "too_slow",
            f"cold run {cold:.0f}s exceeded {cap:.0f}s per-run cap",
        )
    flush(cold, None, [], [], load_sec, "COPASI Direct method" + unit_note, "partial", "")

    warm, cum = [], 0.0
    for i in range(warm_n):
        if cum > warm_budget:
            break
        try:
            w = one(seed_base + 1 + i, first=False)
        except Exception:
            break
        if w > cap:  # this rep overran the per-run cap — don't count it, stop warm
            break
        warm.append(w)
        cum += w
    with contextlib.suppress(Exception):
        COPASI.CRootContainer.removeDatamodel(dm)
    return cold, None, warm, [], load_sec, "COPASI Direct method (exact SSA)" + unit_note, "ok", ""


_RUNNERS = {
    "bngsim": run_bngsim,
    "run_network": run_run_network,
    "roadrunner": run_roadrunner,
    "copasi": run_copasi,
}


def main():
    engine, model_key, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
    out = Path(out_json)
    m = C.MODELS[model_key]
    status_cov, reason = C.cell_status(model_key, engine)
    kind_for_engine, path = C.artifact_for(model_key, engine)
    t_end, n_points, warm_n = float(m["t_end"]), int(m["n_points"]), int(m["warm"])
    import os

    if os.environ.get("SSA_WARM_OVERRIDE"):
        warm_n = int(os.environ["SSA_WARM_OVERRIDE"])
    seed_base = 1000

    base = {
        "model": model_key,
        "engine": engine,
        "kind": m["kind"],
        "kind_for_engine": kind_for_engine,
        "artifact": str(Path(path).name),
        "t_end": t_end,
        "n_points": n_points,
        "warm_N_requested": warm_n,
        "conversion": _conversion_label(model_key, engine),
        "faithfulness_flag": reason if status_cov == "flag" else "",
        "notes": [],
    }

    if status_cov == "na":
        rec = {
            **base,
            "status": "N/A",
            "na_reason": reason,
            "cold_sec": None,
            "warm_median_sec": None,
            "n_warm": 0,
            "events_cold": None,
            "events_warm_median": None,
            "load_sec": None,
            "events_self": False,
            "backend": "",
            "error": "",
        }
        out.write_text(json.dumps(rec, indent=2))
        return

    def flush(cold, ev_cold, warm, ev_warm, load_sec, backend, status, error):
        rec = {
            **base,
            "status": status,
            "load_sec": round(load_sec, 6) if load_sec is not None else None,
            "cold_sec": round(cold, 6) if cold is not None else None,
            "events_cold": ev_cold,
            "warm_median_sec": _warm_median(warm),
            "n_warm": len(warm),
            "warm_secs": [round(x, 6) for x in warm],
            "events_warm_median": (int(st.median(ev_warm)) if ev_warm else None),
            "events_self": ev_cold is not None,
            "backend": backend,
            "error": error,
        }
        out.write_text(json.dumps(rec, indent=2))

    runner = _RUNNERS[engine]
    try:
        cold, ev_cold, warm, ev_warm, load_sec, backend, status, error = runner(
            kind_for_engine,
            path,
            t_end,
            n_points,
            warm_n,
            seed_base,
            C.PER_RUN_CAP_SEC,
            C.WARM_BUDGET_SEC,
            flush,
        )
    except Exception as e:  # noqa: BLE001 — last-resort catch so the cell always writes
        rec = {
            **base,
            "status": "sim_fail",
            "error": f"{type(e).__name__}: {e}"[:300],
            "cold_sec": None,
            "warm_median_sec": None,
            "n_warm": 0,
            "events_cold": None,
            "events_warm_median": None,
            "load_sec": None,
            "events_self": False,
            "backend": "",
        }
        out.write_text(json.dumps(rec, indent=2))
        return

    if status in ("too_slow", "sim_fail", "load_error"):
        rec = {
            **base,
            "status": status,
            "error": error,
            "cold_sec": round(cold, 6) if cold is not None else None,
            "warm_median_sec": None,
            "n_warm": 0,
            "events_cold": ev_cold,
            "events_warm_median": None,
            "load_sec": round(load_sec, 6) if load_sec is not None else None,
            "events_self": ev_cold is not None,
            "backend": backend,
        }
        out.write_text(json.dumps(rec, indent=2))
        return

    note = []
    if len(warm) < warm_n:
        note.append(
            f"warm truncated to {len(warm)}/{warm_n} (per-run cap or {C.WARM_BUDGET_SEC:.0f}s warm budget)"
        )
    base["notes"] = note
    flush(cold, ev_cold, warm, ev_warm, load_sec, backend, "ok", "")


if __name__ == "__main__":
    main()
