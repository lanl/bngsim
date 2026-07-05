#!/usr/bin/env python3
"""GH #78 — MIR-JIT runtime-throughput benchmark (cc vs MIR vs RoadRunner, SBML/ODE).

The decisive gate for whether the MIR codegen backend is worth turning on. The
prototype already proved MIR's *compile latency* wins (~50x small, ~10-18x
multi-MB) and that its generated RHS is numerically bit-identical to cc. What is
unmeasured — and what this settles — is the *runtime throughput* of MIR's JIT'd
code per RHS evaluation vs cc ``-O3`` and vs RoadRunner's LLVM-JIT'd RHS. MIR is
a lightweight JIT, not a full optimizer, so its code is plausibly slower per
step; over millions of steps a 1.5-2x slower RHS would erase the compile win.

Scope: **ODE only, SBML only.** Never flips MIR to default.

Two timing realities on real hardware forced this harness's shape (both verified
directly — see ``dev/notes/gh78_mir_jit_findings.md``):

  1. ``sim.run()`` CONTINUES from the previous end-state (``r2[0] == r1[-1]``), so
     naive repeated runs integrate cheap steady-state continuations, NOT
     representative cold-start integrations. We call ``model.reset()`` before each
     timed run — restoring initial conditions WITHOUT rebuilding the Simulator
     (warm JIT/.so preserved). (RR's analog is ``rr.reset()``.)

  2. A heavy compile burst (cc ``-O3`` subprocess, MIR's c2mir+MIR_gen JIT, RR's
     LLVM) THERMALLY THROTTLES the CPU for several seconds afterward. cc's ~3 s
     compile throttles cc's *own* timing ~5x while MIR's light JIT throttles MIR's
     barely — a spurious asymmetry that made cc look 5x slower than byte-identical
     MIR code. Absolute integrate times drift ~3x with ambient thermal state.
     FIX: time all engines **back-to-back, interleaved, in ONE process**, so every
     engine sees identical thermal conditions each round. The per-round RATIO is
     then thermal-invariant (validated: stable 1.00 even as absolute times drift),
     and the ratio — not the absolute ms — is the headline.

INTEGRATION-ISOLATED basis (this re-run's whole point — supersedes the confounded
first run, see ``dev/notes/gh78_mir_jit_findings.md``). The first run timed a FULL
trajectory (job ``n_points``, up to 1001 output rows); on mid/large models 77-98%
of that wall time is bngsim's INTERPRETED per-output-row evaluation of every
observable + expression (GH #136), which codegen does NOT touch — so it diluted the
RHS signal the gate is about and faked an RR win. The fix: time integration at
``n_points=2`` (CVODE still takes every internal adaptive step over the full
horizon — verified ``n_rhs_evals`` is stable vs full ``n_points`` — but writes only
2 output rows, so the observable/expression cost is ~0). The headline throughput is
the ``n_points=2`` number; the per-output-row cost is measured separately and kept
OUT of the verdict.

Timing decomposition (pre-sim PREP vs the SIMULATION itself):

  * ``load_ms``        — ``Model.from_sbml`` (bngsim) / ``RoadRunner(xml)`` (RR).
                         For cc this is where the ``.so`` compile lands (cold) or a
                         disk-cache hit (flagged); for MIR the C-source generation.
  * ``warm_ms``        — the FIRST run after construction (at ``n_points=2``). For
                         MIR this is where the c2mir+MIR_gen JIT happens (lazy, per
                         Simulator); for cc a cheap dlopen; for RR the first LLVM'd
                         integration. n_points=2 so warm is JIT+integration, not the
                         output cost.
  * ``setup_ms``       — ``load_ms + warm_ms``: total one-time pre-simulation prep.
  * ``integration_ms`` — pure RHS + Jacobian + linear-solve wall time, timed at
                         ``n_points=2`` (median of N warm, reset-to-IC, interleaved
                         runs). THE #78 throughput number — the headline basis.
  * ``output_ms``      — ``full_ms − integration_ms`` where ``full_ms`` is the same
                         integration timed at the job's full ``n_points``. This is
                         the per-output-row observable/expression cost (the GH #136
                         cost). Reported, but NEVER enters the throughput verdict.
  * ``total_ms``       — ``setup_ms + integration_ms + output_ms``: practical wall
                         time for one prep+integrate+output of a full trajectory.

Headline metric: the per-round geomean of ``integration_mir / integration_cc`` (and
``/rr``, ``cc/rr``) on the ``n_points=2`` times — >1 means MIR's RHS (and, on the
dense path, its codegen'd Jacobian) is slower per step than cc ``-O3``. RR is timed
at ``n_points=2`` too so the comparison is matched-output, apples-to-apples. We also
report ``integration_ms / n_rhs_evals`` (per-RHS-eval cost) to normalize out step
count, since ``n_points=2`` gives few steps for fast-equilibrating models.

Correctness gate (in-worker, all engines coexist): cc vs MIR ~0 (byte-identical C
source — the thing #78 puts on trial); bngsim vs RR a loose sanity bound (the
rigorous 1e-4 oracle is owned by the committed rr_parity run at its forced tight
tol — here we integrate at the looser per-job SED-ML tol, where two correct CVODE
runs legitimately diverge, so RR is only a gross-corruption check). Failures are
excluded from geomeans and listed.

Each MODEL runs in its own spawned subprocess (RR can segfault on pathological
SBML — though this corpus is already RR-cross-validated by rr_parity), reusing the
rr_parity ``schedule()`` kill-on-overrun scheduler.

Usage:
    cd bngsim && .venv/bin/python harness/comparison/bench_mir_runtime.py \\
        --stratified 60                              # size-spread sample (serial)
    .venv/bin/python harness/comparison/bench_mir_runtime.py --quick 10
    .venv/bin/python harness/comparison/bench_mir_runtime.py \\
        --models BIOMD0000000001,BIOMD0000000470
    .venv/bin/python harness/comparison/bench_mir_runtime.py --resume
    .venv/bin/python harness/comparison/bench_mir_runtime.py --analyze
    # honest cold cc-compile setup numbers (clears the .so cache once, loud):
    .venv/bin/python harness/comparison/bench_mir_runtime.py --stratified 40 --cold-compile

Output: results/bench_mir_runtime.json (per-model timing + bin-stratified analysis).

NOTE on concurrency: timing is single-process-per-model and the DEFAULT is
``--workers 1`` (serial), because concurrent workers contend for CPU and corrupt
the absolute integrate times. The per-round ratios are robust to this, but the
absolute ms are only meaningful serially.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np

HERE = Path(__file__).resolve().parent
HARNESS_DIR = HERE.parent
SUITE_DIR = (HARNESS_DIR.parent / "parity_checks" / "rr_parity").resolve()
# So the spawned child (spawn re-imports this module) resolves `_rr_common` + `common`.
sys.path.insert(0, str(HARNESS_DIR))
sys.path.insert(0, str(SUITE_DIR))

import _rr_common as rc  # noqa: E402  (align_common + kill-on-overrun scheduler)
from common import RESULTS_DIR, get_machine_info, save_results  # noqa: E402

ODE_JOBS = SUITE_DIR / "ode_jobs.json"
CHECKPOINT = "bench_mir_runtime"
CACHE_DIR = Path.home() / ".cache" / "bngsim" / "codegen"

SIZE_BINS = [
    ("1-10", 1, 10),
    ("11-50", 11, 50),
    ("51-150", 51, 150),
    ("151-500", 151, 500),
    ("501+", 501, 10**9),
]

# Integration-count multipliers for the setup+K*integrate amortization model (a
# fit/scan does 1e2-1e4 integrations of one model).
AMORT_K = [1, 10, 100, 1000, 10000]

DEFAULT_WARMUP = 2  # interleaved warmup ROUNDS (discarded; settle thermal + caches)
DEFAULT_RUNS = 5  # interleaved timed rounds (median)
DEFAULT_ODE_TIMEOUT = 600.0  # per-MODEL wall cap (all engines + big cc compiles)
FINGERPRINT_ROWS = 33
N_POINTS_INTEGRATION = 2  # output rows for the integration-isolated timing (start+end).
# CVODE still takes every internal adaptive step over [t_start, t_end]; only the
# OUTPUT sampling shrinks to 2 rows, so the per-output-row observable/expression
# cost (GH #136) drops out and what remains is pure RHS + Jacobian + linear solve.

ALL_ENGINES = ("cc", "mir", "rr", "exprtk")
DEFAULT_ENGINES = ("cc", "mir", "rr")

CC_VS_MIR_TOL = 1e-4  # hard gate: MIR must reproduce cc's trajectory (it is ~0)
# Loose RR sanity bound: at the per-job SED-ML tol two correct CVODE runs diverge
# more than the tight-tol rr_parity oracle allows; this only catches gross harness
# corruption (wrong horizon/species/NaN). The rigorous 1e-4 vs RR is owned by the
# committed rr_parity run at its forced tight tol.
BN_VS_RR_SANITY = 5e-2


# --------------------------------------------------------------------------- #
# Engine environment + stderr muting
# --------------------------------------------------------------------------- #
def _set_engine_env(engine: str) -> None:
    """Point the SBML loader at ``engine``'s RHS backend. Read live at from_sbml /
    Simulator-construction, so safe to switch between building cc and MIR models in
    the same process (verified). RR ignores all of it."""
    for k in ("BNGSIM_CODEGEN_JIT", "BNGSIM_NO_CODEGEN"):
        os.environ.pop(k, None)
    if engine in ("cc", "mir"):
        os.environ["BNGSIM_CODEGEN_THRESHOLD"] = "1"  # force native RHS on every model
        if engine == "mir":
            os.environ["BNGSIM_CODEGEN_JIT"] = "mir"
    elif engine == "exprtk":
        os.environ["BNGSIM_NO_CODEGEN"] = "1"


class _mute_fd_stderr:
    """Redirect C-level stderr (fd 2) to /dev/null (SUNDIALS/RR chatter)."""

    def __enter__(self):
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        self._saved = os.dup(2)
        os.dup2(self._devnull, 2)
        return self

    def __exit__(self, *exc):
        os.dup2(self._saved, 2)
        os.close(self._saved)
        os.close(self._devnull)
        return False


class _capture_fd_stderr:
    """Redirect C-level stderr (fd 2) to a temp file; ``.text()`` returns its
    contents. Used by the Jacobian-path probe to read ``BNGSIM_JAC_DEBUG`` output,
    which is written at the C level (not via Python's sys.stderr)."""

    def __enter__(self):
        import tempfile

        self._tmp = tempfile.TemporaryFile(mode="w+b")
        self._saved = os.dup(2)
        os.dup2(self._tmp.fileno(), 2)
        return self

    def text(self) -> str:
        with contextlib.suppress(OSError):
            os.fsync(2)
        self._tmp.seek(0)
        return self._tmp.read().decode("utf-8", "replace")

    def __exit__(self, *exc):
        os.dup2(self._saved, 2)
        os.close(self._saved)
        self._tmp.close()
        return False


def _classify_jac_path(jac_stderr: str) -> str:
    """Map BNGSIM_JAC_DEBUG stderr to the linear-solver / Jacobian path.

    The C++ side (cvode_simulator.cpp) prints ``[jac] dense Jacobian: …`` ONLY on
    the dense path; on the sparse KLU path it prints nothing. So:
      * 'compiled (bngsim_codegen_jac)' in stderr → dense, CODEGEN'd Jacobian
        (cc/MIR JIT a giant dense Jac function — the large-model MIR penalty bites).
      * 'interpreted' in stderr → dense, interpreted Jacobian (no codegen).
      * neither → sparse KLU path, INTERPRETED sparse Jacobian (codegen touches
        only the RHS — MIR's penalty there is RHS-only).
    """
    if "compiled (bngsim_codegen_jac)" in jac_stderr:
        return "dense_codegen_jac"
    if "[jac] dense Jacobian: interpreted" in jac_stderr:
        return "dense_interpreted_jac"
    return "sparse_klu_interpreted_jac"


def _jac_worker(spec: dict, q) -> None:
    """Characterize ONE model's linear-solver / Jacobian path (no timing). Runs cc
    codegen (the dense/sparse decision is structural, backend-independent) at
    n_points=2 with BNGSIM_JAC_DEBUG and classifies the path from captured stderr."""
    os.environ.pop("BNGSIM_CODEGEN_JIT", None)
    os.environ.pop("BNGSIM_NO_CODEGEN", None)
    os.environ["BNGSIM_CODEGEN_THRESHOLD"] = "1"
    os.environ["BNGSIM_JAC_DEBUG"] = "1"
    p = spec["params"]
    out = {"key": spec["key"], "model_id": spec["model_id"]}
    try:
        import bngsim

        model = bngsim.Model.from_sbml(spec["xml"])
        sim = bngsim.Simulator(model, method="ode")
        with _capture_fd_stderr() as cap:
            model.reset()
            r = sim.run(
                t_span=(p["t_start"], p["t_end"]),
                n_points=N_POINTS_INTEGRATION,
                rtol=p["rtol"],
                atol=p["atol"],
            )
            jac_txt = cap.text()
        st = {k: int(v) for k, v in dict(r.solver_stats).items()}
        out.update(
            {
                "status": "ok",
                "n_species": int(model.n_species),
                "n_reactions": int(model.n_reactions),
                "jac_path": _classify_jac_path(jac_txt),
                "n_rhs_evals": st.get("n_rhs_evals"),
                "n_jac_evals": st.get("n_jac_evals"),
                "n_steps": st.get("n_steps"),
                "codegen_backend": "mir"
                if getattr(model, "_codegen_c_source", "")
                else ("cc" if getattr(model, "_codegen_so_path", "") else "exprtk"),
            }
        )
    except Exception as exc:  # noqa: BLE001
        out.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"[:200]})
    q.put(out)


def _cache_snapshot() -> set[str]:
    try:
        return {p.name for p in CACHE_DIR.glob("rhs_*") if p.suffix in (".so", ".dylib")}
    except Exception:
        return set()


# --------------------------------------------------------------------------- #
# Fingerprint (compact trajectory for the in-worker correctness gate)
# --------------------------------------------------------------------------- #
def _fingerprint(time_arr, values, names, k: int = FINGERPRINT_ROWS) -> dict:
    t = np.asarray(time_arr)
    v = np.asarray(values)
    n = len(t)
    idx = np.arange(n) if n <= k else np.unique(np.linspace(0, n - 1, k).round().astype(int))
    return {"time": t[idx], "values": v[idx], "names": list(names)}


def _aligned_max_rel(fp_a: dict, fp_b: dict, floor: float = 1e-12) -> float | None:
    """Max relative error over the common species across the fingerprint rows.
    None means time-grid or species-set mismatch (structural — never a silent pass).
    Pairs where both values are below ``floor`` are ignored (rel err meaningless)."""
    ta, tb = np.asarray(fp_a["time"]), np.asarray(fp_b["time"])
    if ta.shape != tb.shape or not np.allclose(ta, tb, rtol=0, atol=1e-6):
        return None
    align = rc.align_common(fp_a["names"], fp_b["names"])
    if align is None:
        return None
    ia, ib, _common = align
    a = np.asarray(fp_a["values"])[:, ia]
    b = np.asarray(fp_b["values"])[:, ib]
    denom = np.maximum(np.abs(a), np.abs(b))
    mask = denom > floor
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(a[mask] - b[mask]) / denom[mask]))


# --------------------------------------------------------------------------- #
# Per-engine build (setup) + one timed integration
# --------------------------------------------------------------------------- #
def _build_bngsim(engine: str, xml: str, run_kw_int: dict, run_kw_full: dict) -> dict:
    """Build a bngsim engine and warm it. Caller must have set the engine env.

    Warms at ``n_points=2`` (``run_kw_int``) — the first run is where MIR JITs and
    cc dlopens, and n_points=2 keeps ``warm_ms`` to JIT + integration without the
    output cost. The *integration* solver stats (``n_rhs_evals`` etc.) come from
    that same n_points=2 run, so the per-RHS-eval normalization uses the pure
    integration step count. A separate full-``n_points`` run (NOT timed into setup)
    produces the correctness fingerprint over the whole trajectory.

    Returns a handle (keeps model+sim for the timed loop) + setup measurements."""
    import bngsim

    cache_before = _cache_snapshot() if engine == "cc" else set()
    t0 = time.perf_counter()
    model = bngsim.Model.from_sbml(xml)
    load_ms = (time.perf_counter() - t0) * 1e3

    has_so = bool(getattr(model, "_codegen_so_path", ""))
    has_cs = bool(getattr(model, "_codegen_c_source", ""))
    backend = "mir" if has_cs else ("cc" if has_so else "exprtk")
    cache_hit = (len(_cache_snapshot() - cache_before) == 0) if engine == "cc" else None

    sim = bngsim.Simulator(model, method="ode")
    model.reset()
    t0 = time.perf_counter()
    r_int = sim.run(**run_kw_int)  # warm: JIT/dlopen happens here, at n_points=2
    warm_ms = (time.perf_counter() - t0) * 1e3
    int_stats = {k: int(v) for k, v in dict(r_int.solver_stats).items()}
    # Full-trajectory run (warm RHS, not timed into setup) for the correctness gate.
    model.reset()
    r_full = sim.run(**run_kw_full)
    full_stats = {k: int(v) for k, v in dict(r_full.solver_stats).items()}
    fp = _fingerprint(r_full.time, r_full.species, r_full.species_names)
    return {
        "kind": "bngsim",
        "model": model,
        "sim": sim,
        "run_kw_int": run_kw_int,
        "run_kw_full": run_kw_full,
        "load_ms": load_ms,
        "warm_ms": warm_ms,
        "backend": backend,
        "cache_hit": cache_hit,
        "n_species": int(model.n_species),
        "solver_stats": int_stats,  # integration (n_points=2) stats — the headline basis
        "full_solver_stats": full_stats,
        "n_steps": int_stats.get("n_steps"),
        "n_rhs_evals": int_stats.get("n_rhs_evals"),
        "n_jac_evals": int_stats.get("n_jac_evals"),
        "fingerprint": fp,
    }


def _build_rr(xml: str, p: dict) -> dict:
    import roadrunner

    roadrunner.Logger.setLevel(roadrunner.Logger.LOG_FATAL)
    t0 = time.perf_counter()
    rr = roadrunner.RoadRunner(xml)
    load_ms = (time.perf_counter() - t0) * 1e3
    rr.integrator = "cvode"
    rr.integrator.relative_tolerance = p["rtol"]
    rr.integrator.absolute_tolerance = p["atol"]
    # Mirror _rr_common.rr_ode's concentration-vs-amount + dropped-boundary handling.
    boundary = list(rr.model.getBoundarySpeciesIds())
    bset = set(boundary)
    sel = [f"[{s}]" if s in bset else s for s in rr.timeCourseSelections]
    present = {s[1:-1] if s.startswith("[") else s for s in sel}
    sel.extend(f"[{s}]" for s in boundary if s not in present)
    rr.timeCourseSelections = sel

    def _names(res):
        cols = [c[1:-1] if c.startswith("[") else c for c in res.colnames]
        return cols[1:]

    # Warm at n_points=2 (LLVM compile + first integration); matched-output basis
    # for the integration comparison. A full-n_points run gives the fingerprint.
    rr.reset()
    t0 = time.perf_counter()
    rr.simulate(p["t_start"], p["t_end"], N_POINTS_INTEGRATION)
    warm_ms = (time.perf_counter() - t0) * 1e3
    rr.reset()
    res = rr.simulate(p["t_start"], p["t_end"], p["n_points"])
    arr = np.asarray(res)
    fp = _fingerprint(arr[:, 0], arr[:, 1:], _names(res))
    return {
        "kind": "rr",
        "rr": rr,
        "p": p,
        "load_ms": load_ms,
        "warm_ms": warm_ms,
        "backend": "roadrunner",
        "cache_hit": None,
        "n_species": int(rr.model.getNumFloatingSpecies()),
        "solver_stats": {},
        "full_solver_stats": {},
        "n_steps": None,
        "n_rhs_evals": None,
        "n_jac_evals": None,
        "fingerprint": fp,
    }


def _time_one(handle: dict, mode: str) -> float:
    """One reset-to-IC integration on a warm engine handle. Returns wall ms.

    ``mode='int'`` integrates at ``n_points=2`` (pure RHS+Jac+solve — the headline);
    ``mode='full'`` at the job's full ``n_points`` (adds the per-output-row
    observable/expression cost). Both reset to IC first so every run does identical
    work on the warm RHS.
    """
    if handle["kind"] == "rr":
        rr, p = handle["rr"], handle["p"]
        npts = N_POINTS_INTEGRATION if mode == "int" else p["n_points"]
        rr.reset()
        t0 = time.perf_counter()
        rr.simulate(p["t_start"], p["t_end"], npts)
    else:
        run_kw = handle["run_kw_int"] if mode == "int" else handle["run_kw_full"]
        handle["model"].reset()
        t0 = time.perf_counter()
        handle["sim"].run(**run_kw)
    return (time.perf_counter() - t0) * 1e3


# --------------------------------------------------------------------------- #
# Worker (one MODEL, all engines interleaved) — module-level for spawn-pickling
# --------------------------------------------------------------------------- #
def _worker(spec: dict, q) -> None:
    """Time all engines for ONE model, interleaved, and put a result on ``q``.

    Setup is sequential (each engine's compile/JIT/LLVM burst, measured); then a
    single continuous cooldown sheds the burst heat; then timed rounds round-robin
    the engines so every round is thermally matched. The per-round ratio is the
    thermal-invariant headline.
    """
    engines = spec["engines"]
    p = spec["params"]
    base_kw = {"rtol": p["rtol"], "atol": p["atol"]}
    if spec.get("max_steps"):
        base_kw["max_steps"] = int(spec["max_steps"])
    run_kw_int = {
        "t_span": (p["t_start"], p["t_end"]),
        "n_points": N_POINTS_INTEGRATION,
        **base_kw,
    }
    run_kw_full = {
        "t_span": (p["t_start"], p["t_end"]),
        "n_points": p["n_points"],
        **base_kw,
    }
    warmup, runs = spec["warmup"], spec["runs"]

    res = {"key": spec["key"], "model_id": spec["model_id"], "engines": {}, "errors": {}}
    handles: dict[str, dict] = {}

    with _mute_fd_stderr():
        # --- SETUP: build + warm each engine (sequential; bursts happen here) ---
        for e in engines:
            try:
                if e == "rr":
                    h = _build_rr(spec["xml"], p)
                else:
                    _set_engine_env(e)
                    h = _build_bngsim(e, spec["xml"], run_kw_int, run_kw_full)
                handles[e] = h
            except Exception as exc:  # noqa: BLE001
                res["errors"][e] = f"{type(exc).__name__}: {exc}"[:300]

        if not handles:
            res["status"] = "error"
            q.put(_strip_handles(res))
            return

        # --- COOLDOWN: one continuous idle to shed the setup burst heat. Sized to
        # the heaviest warm burst; ratios are robust to residual throttle (matched
        # per round), so this mainly de-throttles the ABSOLUTE integrate ms. ---
        worst_warm = max((h["warm_ms"] for h in handles.values()), default=0.0)
        cooldown_s = min(max(worst_warm / 1000.0 * 0.5, 0.5), 12.0)
        time.sleep(cooldown_s)

        # --- TIMED: interleaved rounds (warmup discarded). Each round runs an
        # integration sub-round (n_points=2, the headline — all engines back-to-back
        # so they're tightly thermal-matched) then a full sub-round (job n_points)
        # for the output cost. ---
        order = [e for e in engines if e in handles]
        per_round_int = {e: [] for e in order}
        per_round_full = {e: [] for e in order}
        for rnd in range(warmup + runs):
            # Alternate engine order each round so intra-round core heating doesn't
            # systematically penalize whichever engine always runs last.
            this_order = order if rnd % 2 == 0 else order[::-1]
            for mode, bucket in (("int", per_round_int), ("full", per_round_full)):
                for e in this_order:
                    try:
                        dt = _time_one(handles[e], mode)
                    except Exception as exc:  # noqa: BLE001
                        res["errors"].setdefault(
                            e, f"timed[{mode}]: {type(exc).__name__}: {exc}"[:200]
                        )
                        dt = None
                    if rnd >= warmup and dt is not None:
                        bucket[e].append(dt)

    # --- assemble per-engine timing + correctness + ratios (outside the mute) ---
    n_species = next(
        (handles[e]["n_species"] for e in ("cc", "mir", "exprtk") if e in handles), None
    )
    for e, h in handles.items():
        int_s = per_round_int.get(e, [])
        full_s = per_round_full.get(e, [])
        integration_ms = median(int_s) if int_s else None
        full_ms = median(full_s) if full_s else None
        # output cost is the per-output-row observable/expression overhead (GH #136);
        # clamp tiny-model timing noise to 0 (a 2-row job can't cost more than full).
        output_ms = max(0.0, full_ms - integration_ms) if (full_ms and integration_ms) else None
        setup_ms = h["load_ms"] + h["warm_ms"]
        total_ms = (
            setup_ms + integration_ms + output_ms
            if (integration_ms is not None and output_ms is not None)
            else None
        )
        nre = h["n_rhs_evals"]
        per_rhs_eval_us = (
            (integration_ms * 1e3 / nre) if (integration_ms is not None and nre) else None
        )
        res["engines"][e] = {
            "backend": h["backend"],
            "cache_hit": h["cache_hit"],
            "n_species": h["n_species"],
            "load_ms": round(h["load_ms"], 4),
            "warm_ms": round(h["warm_ms"], 4),
            "setup_ms": round(setup_ms, 4),
            "integration_ms": round(integration_ms, 5) if integration_ms is not None else None,
            "integration_min": round(min(int_s), 5) if int_s else None,
            "integration_samples": [round(s, 5) for s in int_s],
            "full_ms": round(full_ms, 5) if full_ms is not None else None,
            "output_ms": round(output_ms, 5) if output_ms is not None else None,
            "total_ms": round(total_ms, 4) if total_ms is not None else None,
            "per_rhs_eval_us": round(per_rhs_eval_us, 4) if per_rhs_eval_us is not None else None,
            "n_steps": h["n_steps"],
            "n_rhs_evals": h["n_rhs_evals"],
            "n_jac_evals": h.get("n_jac_evals"),
            "solver_stats": h["solver_stats"],
            "full_solver_stats": h.get("full_solver_stats", {}),
        }
    res["n_species"] = n_species
    res["cooldown_s"] = round(cooldown_s, 2)

    # Per-round ratios (thermal-invariant): geomedian over rounds where both timed.
    # Headline ratios are on INTEGRATION (n_points=2); full ratios are reported too
    # so the output-dilution effect (the original confound) is visible.
    res["round_ratios"] = {
        "mir_over_cc": _round_ratios(per_round_int, "mir", "cc"),
        "mir_over_rr": _round_ratios(per_round_int, "mir", "rr"),
        "cc_over_rr": _round_ratios(per_round_int, "cc", "rr"),
        "exprtk_over_cc": _round_ratios(per_round_int, "exprtk", "cc"),
    }
    res["round_ratios_full"] = {
        "mir_over_cc": _round_ratios(per_round_full, "mir", "cc"),
        "mir_over_rr": _round_ratios(per_round_full, "mir", "rr"),
        "cc_over_rr": _round_ratios(per_round_full, "cc", "rr"),
    }

    # Correctness (all engines coexist here).
    corr = {}
    fps = {e: handles[e]["fingerprint"] for e in handles}
    if "cc" in fps and "mir" in fps:
        corr["cc_vs_mir_max_rel"] = _aligned_max_rel(fps["cc"], fps["mir"])
    bn = "cc" if "cc" in fps else ("mir" if "mir" in fps else None)
    if bn and "rr" in fps:
        corr["bngsim_vs_rr_max_rel"] = _aligned_max_rel(fps[bn], fps["rr"])
        corr["bngsim_vs_rr_engine"] = bn
    cm = corr.get("cc_vs_mir_max_rel")
    br = corr.get("bngsim_vs_rr_max_rel")
    corr["gate_cc_mir_ok"] = cm is None or cm < CC_VS_MIR_TOL
    corr["gate_bn_rr_ok"] = br is None or br < BN_VS_RR_SANITY
    corr["gate_ok"] = bool(corr["gate_cc_mir_ok"] and corr["gate_bn_rr_ok"])
    res["correctness"] = corr
    res["status"] = "ok"
    q.put(_strip_handles(res))


def _round_ratios(per_round: dict, num: str, den: str) -> dict | None:
    """Median + samples of per-round num/den ratios (thermal-matched per round)."""
    a, b = per_round.get(num, []), per_round.get(den, [])
    n = min(len(a), len(b))
    if n == 0:
        return None
    ratios = [a[i] / b[i] for i in range(n) if b[i] > 0]
    if not ratios:
        return None
    return {"median": round(median(ratios), 4), "samples": [round(x, 4) for x in ratios]}


def _strip_handles(res: dict) -> dict:
    """Drop any live engine handles / numpy fingerprints before queueing."""
    return res


# --------------------------------------------------------------------------- #
# Job loading / sampling
# --------------------------------------------------------------------------- #
def _load_jobs() -> list[dict]:
    return json.loads(ODE_JOBS.read_text())["jobs"]


def _xml_path(job: dict) -> Path:
    return (SUITE_DIR / job["model"]).resolve()


def _select_jobs(jobs: list[dict], args) -> list[dict]:
    out = [j for j in jobs if _xml_path(j).exists()]
    if args.models:
        wanted = {x.strip() for x in args.models.split(",") if x.strip()}
        out = [j for j in out if j["model_id"] in wanted]
    if args.include:
        out = [j for j in out if args.include in j["model"]]
    if args.exclude:
        out = [j for j in out if args.exclude not in j["model"]]
    if args.stratified:
        out = _stratified_by_size(out, args.stratified)
    elif args.quick:
        out = out[: args.quick]
    if args.limit:
        out = out[: args.limit]
    return out


def _stratified_by_size(jobs: list[dict], n: int) -> list[dict]:
    """``n`` jobs spread across the SBML byte-size range (proxy for model size,
    since n_species is unknown until load). Deterministic even-stride over the
    size-sorted list."""
    if n >= len(jobs):
        return jobs
    sized = sorted(jobs, key=lambda j: _xml_path(j).stat().st_size)
    step = len(sized) / n
    return [sized[min(len(sized) - 1, int(i * step))] for i in range(n)]


# --------------------------------------------------------------------------- #
# Checkpoint
# --------------------------------------------------------------------------- #
def _ckpt_path() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{CHECKPOINT}_timing.json"


def _load_checkpoint() -> dict:
    p = _ckpt_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _save_checkpoint(done: dict) -> None:
    _ckpt_path().write_text(json.dumps(done, default=_json_default))


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def geomean(values) -> float:
    vals = [v for v in values if v is not None and v > 0 and math.isfinite(v)]
    if not vals:
        return 0.0
    return float(math.exp(sum(math.log(v) for v in vals) / len(vals)))


def _ratio_med(m: dict, key: str, *, full: bool = False):
    rr = m.get("round_ratios_full" if full else "round_ratios", {}).get(key)
    return rr["median"] if rr else None


def analyze(models: list[dict]) -> dict:
    gated = [m for m in models if m.get("correctness", {}).get("gate_ok")]

    def bin_rows(key, *, full: bool = False):
        rows = []
        for label, lo, hi in SIZE_BINS:
            bm = [m for m in gated if lo <= (m.get("n_species") or 0) <= hi]
            rows.append(
                {
                    "bin": label,
                    "n": len(bm),
                    "geomean": geomean(_ratio_med(m, key, full=full) for m in bm),
                }
            )
        return rows

    overall = {
        "n_models": len(models),
        "n_gated_ok": len(gated),
        # headline = INTEGRATION-isolated (n_points=2) ratios
        "mir_over_cc_geomean": geomean(_ratio_med(m, "mir_over_cc") for m in gated),
        "mir_over_rr_geomean": geomean(_ratio_med(m, "mir_over_rr") for m in gated),
        "cc_over_rr_geomean": geomean(_ratio_med(m, "cc_over_rr") for m in gated),
        "mir_integrate_wins_vs_cc": sum(
            1 for m in gated if (_ratio_med(m, "mir_over_cc") or 9) < 1.0
        ),
        "mir_within_5pct_of_cc": sum(
            1 for m in gated if abs((_ratio_med(m, "mir_over_cc") or 9) - 1.0) <= 0.05
        ),
        # full-trajectory ratios (output-diluted) — shown only to expose the original
        # confound: the per-output-row observable cost is COMMON-MODE, so it pulls
        # every ratio toward 1.0 and toward RR.
        "mir_over_cc_geomean_FULL": geomean(
            _ratio_med(m, "mir_over_cc", full=True) for m in gated
        ),
        "mir_over_rr_geomean_FULL": geomean(
            _ratio_med(m, "mir_over_rr", full=True) for m in gated
        ),
        "cc_over_rr_geomean_FULL": geomean(_ratio_med(m, "cc_over_rr", full=True) for m in gated),
    }
    return {
        "overall": overall,
        "bins": {
            "mir_over_cc": bin_rows("mir_over_cc"),
            "mir_over_rr": bin_rows("mir_over_rr"),
            "cc_over_rr": bin_rows("cc_over_rr"),
            "mir_over_cc_FULL": bin_rows("mir_over_cc", full=True),
            "cc_over_rr_FULL": bin_rows("cc_over_rr", full=True),
        },
        "setup": _setup_summary(gated),
        "output_cost": _output_summary(gated),
        "per_rhs_eval": _per_rhs_table(gated),
        "amortization": _amortization(gated),
        "gate_failures": [
            {
                "model_id": m["model_id"],
                "n_species": m.get("n_species"),
                "cc_vs_mir_max_rel": m.get("correctness", {}).get("cc_vs_mir_max_rel"),
                "bngsim_vs_rr_max_rel": m.get("correctness", {}).get("bngsim_vs_rr_max_rel"),
            }
            for m in models
            if not m.get("correctness", {}).get("gate_ok")
        ],
        "engine_errors": [
            {"model_id": m["model_id"], "engine": e, "error": err}
            for m in models
            for e, err in m.get("errors", {}).items()
        ],
    }


def _setup_summary(models: list[dict]) -> dict:
    """Geomean of setup_ms ratios (the compile-latency story, prep half of the
    breakdown). cc cold-compile vs MIR JIT vs RR LLVM."""

    def setup_ratio(m, a, b):
        ea, eb = m["engines"].get(a), m["engines"].get(b)
        if ea and eb and ea.get("setup_ms") and eb.get("setup_ms"):
            return ea["setup_ms"] / eb["setup_ms"]
        return None

    return {
        "mir_over_cc_geomean": geomean(setup_ratio(m, "mir", "cc") for m in models),
        "mir_over_rr_geomean": geomean(setup_ratio(m, "mir", "rr") for m in models),
        "cc_over_rr_geomean": geomean(setup_ratio(m, "cc", "rr") for m in models),
        "note": (
            "setup_ms = load + warm (first-run JIT/dlopen). cc cold-compile vs MIR "
            "in-process JIT vs RR LLVM. For large models from_sbml is dominated by "
            "the SHARED Python C-source generation (cc and MIR both pay it), which "
            "masks the cc-subprocess compile — the isolated compile-latency win is "
            "in gh78_mir_jit_findings.md's latency table. cc setup is a cold compile "
            "only when cache_hit is False (see --cold-compile)."
        ),
    }


def _output_summary(models: list[dict]) -> dict:
    """How much of the full-trajectory wall time is the per-output-row observable/
    expression cost (GH #136) — the dilution the original run mistook for RHS time.
    Reported as geomean output_ms/full_ms per size bin (bngsim cc engine)."""

    def out_frac(m):
        cc = m["engines"].get("cc") or m["engines"].get("mir")
        if cc and cc.get("full_ms") and cc.get("output_ms") is not None and cc["full_ms"] > 0:
            return cc["output_ms"] / cc["full_ms"]
        return None

    rows = []
    for label, lo, hi in SIZE_BINS:
        bm = [m for m in models if lo <= (m.get("n_species") or 0) <= hi]
        fracs = [f for f in (out_frac(m) for m in bm) if f is not None]
        rows.append(
            {
                "bin": label,
                "n": len(fracs),
                "output_frac_geomean": geomean(fracs) if fracs else 0.0,
            }
        )
    return {
        "by_bin": rows,
        "note": (
            "output_frac = output_ms / full_ms (bngsim). The per-output-row "
            "observable+expression eval (interpreted; GH #136) that codegen does NOT "
            "touch. A high fraction is exactly what diluted the first run's "
            "integrate_ms — here it is measured and excluded from the throughput "
            "verdict, which uses integration_ms (n_points=2) only."
        ),
    }


def _per_rhs_table(models: list[dict]) -> dict:
    """integration_ms / n_rhs_evals (per-RHS-eval cost) for the integration-dominated
    models, normalizing out step count — the cleanest MIR-vs-cc signal."""
    rows = []
    for m in sorted(models, key=lambda m: (m.get("n_species") or 0)):
        cc, mir = m["engines"].get("cc"), m["engines"].get("mir")
        if not (cc and mir):
            continue
        nre = cc.get("n_rhs_evals")
        rows.append(
            {
                "model_id": m["model_id"],
                "n_species": m.get("n_species"),
                "n_rhs_evals": nre,
                "n_steps": cc.get("n_steps"),
                "n_jac_evals": cc.get("n_jac_evals"),
                "cc_integration_ms": cc.get("integration_ms"),
                "cc_per_rhs_eval_us": cc.get("per_rhs_eval_us"),
                "mir_per_rhs_eval_us": mir.get("per_rhs_eval_us"),
                "mir_over_cc": _ratio_med(m, "mir_over_cc"),
            }
        )
    return {
        "models": rows,
        "note": "per_rhs_eval_us = integration_ms*1e3 / n_rhs_evals (n_points=2).",
    }


def _amortization(models: list[dict]) -> dict:
    """setup + K*integration_ms amortization on the PURE-integration basis (the
    throughput question). Practical wall time would add the common-mode output_ms,
    which only pulls the MIR/cc total closer to 1.0, so this is the pessimistic
    (most MIR-unfavorable) amortization."""
    out = {"K_values": AMORT_K, "by_K": []}
    for K in AMORT_K:
        ratios, mir_wins, n = [], 0, 0
        for m in models:
            cc, mir = m["engines"].get("cc"), m["engines"].get("mir")
            if not (cc and mir and cc.get("integration_ms") and mir.get("integration_ms")):
                continue
            cc_total = cc["setup_ms"] + K * cc["integration_ms"]
            mir_total = mir["setup_ms"] + K * mir["integration_ms"]
            if cc_total > 0:
                ratios.append(mir_total / cc_total)
                mir_wins += int(mir_total < cc_total)
                n += 1
        out["by_K"].append(
            {"K": K, "n": n, "mir_over_cc_total_geomean": geomean(ratios), "mir_wins": mir_wins}
        )
    out["note"] = (
        "K = integrations of one model; total = setup + K*integration_ms (PURE "
        "integration, n_points=2). mir_over_cc_total < 1 means MIR's fast-compile + "
        "near-equal-run wins total wall time at that K. Only meaningful when cc setup "
        "is a TRUE compile (cache_hit False / --cold-compile); on a warm cc cache the "
        ".so load is a cheap dlopen and the setup gap vanishes. Adding the common-mode "
        "output_ms (practical full trajectories) only moves the ratio toward 1.0."
    )
    return out


def _print_summary(models: list[dict], analysis: dict) -> None:
    o = analysis["overall"]
    print("\n" + "=" * 80)
    print("  GH #78 MIR runtime throughput — SBML ODE corpus (interleaved per-round ratios)")
    print("=" * 80)
    print(
        f"  models: {o['n_models']}   gate-OK: {o['n_gated_ok']}   "
        f"errors: {len(analysis['engine_errors'])}   gate-fails: {len(analysis['gate_failures'])}"
    )
    print("\n  INTEGRATION-ISOLATED (n_points=2) per-round geomean ratios — >1 = slower per step:")
    print(
        f"    MIR / cc : {o['mir_over_cc_geomean']:.4f}x   "
        f"(MIR integration wins {o['mir_integrate_wins_vs_cc']}/{o['n_gated_ok']}, "
        f"within 5% of cc: {o['mir_within_5pct_of_cc']}/{o['n_gated_ok']})"
    )
    print(f"    MIR / RR : {o['mir_over_rr_geomean']:.4f}x")
    print(f"    cc  / RR : {o['cc_over_rr_geomean']:.4f}x")
    print(
        "  [contrast] FULL-trajectory (output-diluted, the original confound): "
        f"MIR/cc {o['mir_over_cc_geomean_FULL']:.4f}x  cc/RR {o['cc_over_rr_geomean_FULL']:.4f}x"
    )
    s = analysis["setup"]
    print(
        f"\n  SETUP geomean:  MIR/cc {s['mir_over_cc_geomean']:.3f}x   "
        f"MIR/RR {s['mir_over_rr_geomean']:.3f}x   cc/RR {s['cc_over_rr_geomean']:.3f}x"
    )
    print(
        f"\n  {'bin':>9s} {'N':>4s} {'MIR/cc':>9s} {'MIR/RR':>9s} {'cc/RR':>9s}  {'cc/RR(full)':>11s} {'out%':>5s}"
    )
    print("  " + "-" * 66)
    bm = {r["bin"]: r for r in analysis["bins"]["mir_over_cc"]}
    of = {r["bin"]: r for r in analysis["output_cost"]["by_bin"]}
    for label, _lo, _hi in SIZE_BINS:
        n = bm[label]["n"]
        if not n:
            continue

        def g(key, label=label):
            return next(r["geomean"] for r in analysis["bins"][key] if r["bin"] == label)

        print(
            f"  {label:>9s} {n:>4d} {g('mir_over_cc'):>8.4f}x "
            f"{g('mir_over_rr'):>8.4f}x {g('cc_over_rr'):>8.4f}x  "
            f"{g('cc_over_rr_FULL'):>10.4f}x {100 * of[label]['output_frac_geomean']:>4.0f}%"
        )
    print("\n  PER-RHS-EVAL cost (integration_ms/n_rhs_evals, ns; integration-dominated models):")
    print(
        f"    {'model':>20s} {'nsp':>5s} {'rhs_ev':>7s} {'cc ns':>9s} {'MIR ns':>9s} {'MIR/cc':>7s}"
    )
    for r in analysis["per_rhs_eval"]["models"]:
        if not r.get("n_rhs_evals") or r["n_rhs_evals"] < 50:
            continue  # too few steps → per-RHS noise dominates
        ccn = r.get("cc_per_rhs_eval_us")
        mirn = r.get("mir_per_rhs_eval_us")
        print(
            f"    {r['model_id']:>20s} {r['n_species'] or 0:>5d} {r['n_rhs_evals']:>7d} "
            f"{(ccn * 1e3 if ccn else 0):>9.1f} {(mirn * 1e3 if mirn else 0):>9.1f} "
            f"{r['mir_over_cc'] if r['mir_over_cc'] else 0:>7.3f}"
        )
    print(
        "\n  AMORTIZATION  setup + K*integration  (MIR total / cc total, pure-integration basis):"
    )
    for row in analysis["amortization"]["by_K"]:
        if row["n"]:
            print(
                f"    K={row['K']:>6d}: {row['mir_over_cc_total_geomean']:.3f}x   "
                f"(MIR total wins {row['mir_wins']}/{row['n']})"
            )
    if analysis["gate_failures"]:
        print(f"\n  gate failures ({len(analysis['gate_failures'])}):")
        for f in analysis["gate_failures"][:12]:
            print(
                f"    {f['model_id']} ({f['n_species']}sp): cc~mir={f['cc_vs_mir_max_rel']} "
                f"bn~rr={f['bngsim_vs_rr_max_rel']}"
            )
    print("=" * 80)


# --------------------------------------------------------------------------- #
# Jacobian-path probe (characterization, no timing)
# --------------------------------------------------------------------------- #
def _run_jac_probe(jobs: list[dict], args) -> int:
    specs = [
        {
            "key": j["model_id"],
            "model_id": j["model_id"],
            "xml": str(_xml_path(j)),
            "params": {
                "t_start": float(j["params"]["t_start"]),
                "t_end": float(j["params"]["t_end"]),
                "rtol": float(j["params"]["rtol"]),
                "atol": float(j["params"]["atol"]),
            },
            "cap": args.timeout,
        }
        for j in jobs
    ]
    print("=" * 80)
    print("  GH #78 — Jacobian-path characterization (dense codegen'd vs sparse KLU)")
    print("=" * 80)
    print(
        f"  models: {len(specs)}   (dense path → Jac is codegen'd; sparse KLU → Jac interpreted)\n"
    )

    out: dict[str, dict] = {}

    def _od(finished, total, res):
        out[res["key"]] = res
        if res.get("status") == "ok":
            tag = {
                "dense_codegen_jac": "DENSE  (Jac CODEGEN'd)",
                "dense_interpreted_jac": "DENSE  (Jac interpreted)",
                "sparse_klu_interpreted_jac": "SPARSE-KLU (Jac interpreted)",
            }.get(res["jac_path"], res["jac_path"])
            extra = (
                f"{res.get('n_species')}sp  {tag}  "
                f"rhs_ev={res.get('n_rhs_evals')} jac_ev={res.get('n_jac_evals')}"
            )
        else:
            extra = "ERR " + str(res.get("error"))[:80]
        print(f"  [{finished}/{total}] {res['model_id']:<20s} {extra}", flush=True)

    rc.schedule(specs, _jac_worker, workers=1, timeout_of=lambda s: float(s["cap"]), on_done=_od)

    rows = sorted(out.values(), key=lambda r: (r.get("n_species") or 0, r["model_id"]))
    path = RESULTS_DIR / f"{CHECKPOINT}_jacpath.json"
    path.write_text(json.dumps({"models": rows}, default=_json_default, indent=1))
    # Summary: count by path, and which is dominant for the large models.
    from collections import Counter

    counts = Counter(r.get("jac_path") for r in rows if r.get("status") == "ok")
    print("\n  path counts:", dict(counts))
    big = [r for r in rows if (r.get("n_species") or 0) >= 150 and r.get("status") == "ok"]
    if big:
        print("  large models (≥150 sp):")
        for r in big:
            print(f"    {r['model_id']:>20s} {r['n_species']:>5d}sp  {r['jac_path']}")
    print(f"\n  wrote {path}")
    return 0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--workers", type=int, default=1, help="Concurrent MODEL subprocesses (1=accurate)."
    )
    ap.add_argument("--quick", type=int, default=0, help="First N models (manifest order).")
    ap.add_argument("--stratified", type=int, default=0, help="N models spread across SBML size.")
    ap.add_argument("--limit", type=int, default=0, help="Hard cap after other filters.")
    ap.add_argument("--models", default="", help="Comma-separated model_id filter.")
    ap.add_argument("--include", default="", help="Substring filter on the model path.")
    ap.add_argument("--exclude", default="", help="Substring filter — drop matching paths.")
    ap.add_argument(
        "--engines",
        default=",".join(DEFAULT_ENGINES),
        help=f"Comma list from {ALL_ENGINES} (default {','.join(DEFAULT_ENGINES)}).",
    )
    ap.add_argument(
        "--warmup", type=int, default=DEFAULT_WARMUP, help="Interleaved warmup rounds."
    )
    ap.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS, help="Interleaved timed rounds (median)."
    )
    ap.add_argument(
        "--max-steps", type=int, default=0, help="Bound CVODE steps (0=engine default)."
    )
    ap.add_argument(
        "--timeout", type=float, default=DEFAULT_ODE_TIMEOUT, help="Per-model wall cap (s)."
    )
    ap.add_argument(
        "--cold-compile",
        action="store_true",
        help="Clear the cc .so cache ONCE at start so every cc setup is a TRUE compile "
        "(apples-to-apples with MIR's always-fresh JIT). Loud; clears a regenerable cache.",
    )
    ap.add_argument("--resume", action="store_true", help="Skip models already in the checkpoint.")
    ap.add_argument("--analyze", action="store_true", help="Re-analyze the checkpoint; no timing.")
    ap.add_argument(
        "--jac-probe",
        action="store_true",
        help="Characterize each selected model's linear-solver/Jacobian path "
        "(dense codegen'd vs sparse KLU interpreted); no timing. Writes results/"
        f"{CHECKPOINT}_jacpath.json.",
    )
    ap.add_argument("--out", default=CHECKPOINT, help="Output results basename.")
    args = ap.parse_args()

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    bad = [e for e in engines if e not in ALL_ENGINES]
    if bad:
        sys.exit(f"unknown engine(s): {bad}; choose from {ALL_ENGINES}")

    if args.analyze:
        done = _load_checkpoint()
        if not done:
            sys.exit("no checkpoint to analyze.")
        models = sorted(done.values(), key=lambda m: (m.get("n_species") or 0, m["model_id"]))
        analysis = analyze(models)
        _print_summary(models, analysis)
        _write_final(models, analysis, args, engines)
        return 0

    if not ODE_JOBS.exists():
        sys.exit(f"missing {ODE_JOBS}")
    jobs = _select_jobs(_load_jobs(), args)
    if not jobs:
        sys.exit("no jobs after filtering.")

    if args.jac_probe:
        return _run_jac_probe(jobs, args)

    if args.cold_compile:
        n = 0
        for p in CACHE_DIR.glob("rhs_*"):
            try:
                p.unlink()
                n += 1
            except Exception:
                pass
        print(f"  [--cold-compile] cleared {n} cached codegen artifact(s) from {CACHE_DIR}")

    specs = []
    for j in jobs:
        p = j["params"]
        specs.append(
            {
                "key": j["model_id"],
                "model_id": j["model_id"],
                "engines": engines,
                "xml": str(_xml_path(j)),
                "params": {
                    "t_start": float(p["t_start"]),
                    "t_end": float(p["t_end"]),
                    "n_points": int(p["n_points"]),
                    "rtol": float(p["rtol"]),
                    "atol": float(p["atol"]),
                },
                "warmup": args.warmup,
                "runs": args.runs,
                "max_steps": args.max_steps,
                "cap": args.timeout,
            }
        )

    done = _load_checkpoint() if args.resume else {}
    todo = [s for s in specs if s["key"] not in done]

    print("=" * 80)
    print("  GH #78 — MIR-JIT runtime throughput (cc vs MIR vs RoadRunner)")
    print("=" * 80)
    print(f"  models: {len(jobs)}   engines: {engines}   todo: {len(todo)}   (done: {len(done)})")
    print(
        f"  protocol: integration-isolated — integration at n_points={N_POINTS_INTEGRATION}, "
        f"full at job n_points; interleaved {args.warmup}w+{args.runs}t rounds, reset() each run"
    )
    print(
        f"  workers: {args.workers}{'  (WARNING: >1 corrupts absolute ms)' if args.workers > 1 else ''}"
        f"   wall cap: {args.timeout}s   cold-compile: {args.cold_compile}"
    )
    print()

    counter = {"n": len(done)}

    def _on_done(finished, total, res):
        done[res["key"]] = res
        counter["n"] += 1
        if res.get("status") == "ok":
            r = _ratio_med(res, "mir_over_cc")
            cc = res["engines"].get("cc", {})
            extra = (
                f"{res.get('n_species')}sp  MIR/cc={r if r is not None else 'NA'}  "
                f"cc int={cc.get('integration_ms')}ms out={cc.get('output_ms')}ms  "
                f"gate={'OK' if res.get('correctness', {}).get('gate_ok') else 'FAIL'}"
            )
        else:
            extra = "ALL ENGINES FAILED " + str(res.get("errors"))[:80]
        errs = res.get("errors") or {}
        if errs and res.get("status") == "ok":
            extra += f"  [errs: {list(errs)}]"
        print(f"  [{finished}/{total}] {res['model_id']:<20s} {extra}", flush=True)
        if counter["n"] % 5 == 0:
            _save_checkpoint(done)

    t0 = time.perf_counter()
    if todo:
        rc.schedule(
            todo,
            _worker,
            workers=args.workers,
            timeout_of=lambda s: float(s["cap"]),
            on_done=_on_done,
        )
    elapsed = time.perf_counter() - t0
    _save_checkpoint(done)

    models = sorted(done.values(), key=lambda m: (m.get("n_species") or 0, m["model_id"]))
    analysis = analyze(models)
    analysis["elapsed_sec"] = round(elapsed, 1)
    _print_summary(models, analysis)
    _write_final(models, analysis, args, engines)
    return 0


def _write_final(models, analysis, args, engines) -> None:
    output = {
        "machine_info": get_machine_info(),
        "protocol": {
            "design": "per-model, all engines interleaved round-robin, thermal-matched per-round ratios",
            "basis": "INTEGRATION-ISOLATED: headline ratios on integration_ms timed at "
            f"n_points={N_POINTS_INTEGRATION} (RHS+Jac+solve only). full_ms at the job's "
            "n_points; output_ms = full_ms - integration_ms (the GH #136 per-output-row "
            "observable/expression cost, reported but excluded from the verdict).",
            "n_points_integration": N_POINTS_INTEGRATION,
            "warmup_rounds": args.warmup,
            "timed_rounds": args.runs,
            "max_steps": args.max_steps,
            "engines": engines,
            "cold_compile": args.cold_compile,
            "workers": args.workers,
            "tol_source": "per-job SED-ML rtol/atol from ode_jobs.json (verbatim)",
            "reset_between_runs": "model.reset() / rr.reset() — every timed run from IC on the warm RHS",
            "headline_metric": "per-round geomean integration_mir/integration_cc at n_points=2 "
            "(thermal-invariant); RR matched at n_points=2.",
            "gate": f"cc~mir < {CC_VS_MIR_TOL} (hard); bngsim~RR < {BN_VS_RR_SANITY} (loose sanity; rigorous 1e-4 owned by rr_parity)",
        },
        "analysis": analysis,
        "per_model": models,
    }
    save_results(output, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
