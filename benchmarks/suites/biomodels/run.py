#!/usr/bin/env python3
"""
Stage-2 BioModels suite runner.

Takes every ``keep``-verdict model from ``manifest.csv`` (the Stage-1
structural partition written by ``filter.py``) and, for each one, loads
and simulates it under three ODE engines -- **BNGsim**, **libRoadRunner**
and **AMICI** -- recording per-engine *coverage* (did it load? did it
simulate to a finite trajectory?) and *accuracy* (does the trajectory
agree with the libRoadRunner reference?).

Each engine runs in its own subprocess with a wall-clock timeout, so a
segfault, a hang or a runaway integration in one engine on one model
cannot take down the sweep -- it is recorded as a crash / timeout and
the run moves on.

The events tag from the manifest splits the output: events-tagged
models feed the SBML-events figure (Fig S2), the rest feed the
BioModels figure (Fig S1). Both partitions are written, plus a combined
``coverage.csv``.

Coverage is deliberately kept distinct from the Stage-1 partition: a
``keep`` model that no engine can simulate is a *coverage* miss, not a
filter error -- the filter only promises structural simulability.

Usage:
    # full sweep over every keep model
    python run.py

    # cheap subset (small models only)
    python run.py --effort low

    # one model, verbose
    python run.py --model BIOMD0000000001 -v

    # (internal) single-engine worker -- spawned by the parent
    python run.py --worker --engine roadrunner --sbml PATH --out NPZ ...
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_ROOT = SCRIPT_DIR.parents[1]  # bngsim/benchmarks
sys.path.insert(0, str(BENCH_ROOT))
from _effort import add_effort_arg, filter_by_effort  # noqa: E402

DEFAULT_MANIFEST = SCRIPT_DIR / "manifest.csv"
DEFAULT_SBML_DIR = SCRIPT_DIR / "data" / "sbml_downloads"
RESULTS_DIR = SCRIPT_DIR / "results"

ENGINES = ("roadrunner", "bngsim", "amici")  # roadrunner first -- it is the reference

N_POINTS = 101  # shared output grid -- identical timepoints => no interpolation
T_CANDIDATES = (10.0, 100.0, 1000.0, 10000.0)  # adaptive steady-state horizons
T_FALLBACK = 100.0  # horizon for models that never reach steady state
ACC_RTOL = 1e-3  # max relative error for an accuracy PASS vs the RR reference
ACC_ATOL = 1e-9  # absolute floor for the relative-error denominator
ACC_REL_FLOOR = 1e-6  # per-species denominator floor, as a fraction of model scale
DEFAULT_TIMEOUT = 300.0  # per-engine wall-clock budget (seconds)

# Matched solver tolerances across all three engines -- without this the
# accuracy comparison would measure each engine's *default* tolerance,
# not its correctness. Tight enough that the residual is the integrator
# scheme, not the error control.
SOLVER_RTOL = 1e-8
SOLVER_ATOL = 1e-12
SOLVER_MAX_STEPS = 100_000

# Effort tiers by structural size (n_species + n_reactions). The cutoffs
# are coarse cost proxies; a measured-walltime calibration can refine
# them later (see _effort.py and project_benchmark_effort_tiers).
EFFORT_LOW_MAX = 20
EFFORT_MEDIUM_MAX = 80


def size_tier(n_species: int, n_reactions: int) -> str:
    size = n_species + n_reactions
    if size <= EFFORT_LOW_MAX:
        return "low"
    if size <= EFFORT_MEDIUM_MAX:
        return "medium"
    return "high"


# ==========================================================================
# worker -- runs ONE engine on ONE model, in its own subprocess
# ==========================================================================
def _save(out: Path, **kw) -> None:
    """Write the worker result npz (always written, even on failure)."""
    np.savez(out, **kw)


def _new_rr(sbml: str):
    """A freshly loaded RoadRunner with the matched solver tolerances."""
    import roadrunner

    rr = roadrunner.RoadRunner(sbml)
    integ = rr.getIntegrator()
    integ.setValue("relative_tolerance", SOLVER_RTOL)
    integ.setValue("absolute_tolerance", SOLVER_ATOL)
    integ.setValue("maximum_num_steps", SOLVER_MAX_STEPS)
    return rr


def _phased(load_fn, sim_fn) -> dict:
    """Run a *load* phase then a *sim* phase, recording how far it got.

    ``load_fn()`` returns an opaque context; ``sim_fn(context)`` returns
    a dict of ``time`` / ``names`` / ``data`` (+ optional ``t_end`` /
    ``is_steady``). Splitting the two means a loader failure and a
    solver failure are distinct coverage outcomes -- the whole point of
    a per-engine coverage table -- rather than a single opaque miss."""
    res = dict(load_ok=0, sim_ok=0, load_s=0.0, sim_s=0.0, t_end=0.0, is_steady=0, error="")
    t0 = time.perf_counter()
    try:
        ctx = load_fn()
    except Exception as exc:
        res["error"] = f"load: {type(exc).__name__}: {str(exc)[:180]}"
        return res
    res["load_ok"], res["load_s"] = 1, time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        out = sim_fn(ctx)
    except Exception as exc:
        res["error"] = f"sim: {type(exc).__name__}: {str(exc)[:180]}"
        return res
    res["sim_s"] = time.perf_counter() - t0
    res["sim_ok"] = 1
    res.update(out)
    return res


def worker_roadrunner(sbml: str, t_end, n_points: int) -> dict:
    """libRoadRunner -- the reference engine. ``t_end='auto'`` triggers
    an adaptive steady-state horizon search."""

    def sim(rr):
        is_steady = False
        end = t_end
        if end == "auto":
            # First horizon at which the trajectory tail is flat wins;
            # an oscillatory / never-settling model falls back to a
            # moderate horizon so the engines stay phase-comparable.
            # Each candidate runs on its own fresh RoadRunner -- reset()
            # does not restore event trigger state, so a model with
            # events would carry already-fired events into the next
            # horizon.
            end = T_FALLBACK
            for cand in T_CANDIDATES:
                data = np.asarray(_new_rr(sbml).simulate(0, cand, n_points))[:, 1:]
                tail = data[max(0, len(data) - len(data) // 10) :]
                if len(tail) < 2:
                    continue
                rel = np.ptp(tail, axis=0) / np.maximum(np.max(np.abs(data), axis=0), 1e-10)
                if np.all(rel < 0.01):
                    end, is_steady = cand, True
                    break
            rr = _new_rr(sbml)  # fresh runner for the reference sim
        else:
            end = float(end)
        out = rr.simulate(0, end, n_points)
        arr = np.asarray(out)
        return dict(
            t_end=end,
            is_steady=int(is_steady),
            time=arr[:, 0],
            names=np.array([c.strip("[]") for c in out.colnames[1:]]),
            data=arr[:, 1:],
        )

    return _phased(lambda: _new_rr(sbml), sim)


def worker_bngsim(sbml: str, t_end, n_points: int) -> dict:
    def load():
        import bngsim

        return bngsim.Model.from_sbml(sbml)

    def sim(model):
        import bngsim

        res = bngsim.Simulator(model, method="ode").run(
            t_span=(0.0, float(t_end)),
            n_points=n_points,
            rtol=SOLVER_RTOL,
            atol=SOLVER_ATOL,
            max_steps=SOLVER_MAX_STEPS,
        )
        return dict(
            t_end=float(t_end),
            time=np.asarray(res.time),
            names=np.array(list(model.species_names)),
            data=np.asarray(res.species),
        )

    return _phased(load, sim)


def worker_amici(sbml: str, t_end, n_points: int) -> dict:
    model_id = Path(sbml).stem
    holder: dict = {}

    def load():
        # AMICI compiles the SBML to a C++ extension -- counted as load.
        import amici

        outdir = tempfile.mkdtemp(prefix="amici_")
        holder["outdir"] = outdir
        amici.SbmlImporter(sbml).sbml2amici(model_id, outdir, verbose=False)
        return amici.import_model_module(model_id, outdir).get_model()

    def sim(amodel):
        amodel.set_timepoints(np.linspace(0.0, float(t_end), n_points))
        solver = amodel.create_solver()
        solver.set_relative_tolerance(SOLVER_RTOL)
        solver.set_absolute_tolerance(SOLVER_ATOL)
        solver.set_max_steps(SOLVER_MAX_STEPS)
        rdata = amodel.simulate(solver=solver)
        if rdata.status != 0:
            raise RuntimeError(f"amici solver status {rdata.status}")
        return dict(
            t_end=float(t_end),
            time=np.asarray(rdata.ts),
            names=np.array(list(amodel.get_state_ids())),
            data=np.asarray(rdata.x),
        )

    try:
        return _phased(load, sim)
    finally:
        outdir = holder.get("outdir")
        if outdir:
            shutil.rmtree(outdir, ignore_errors=True)


_WORKERS = {
    "roadrunner": worker_roadrunner,
    "bngsim": worker_bngsim,
    "amici": worker_amici,
}


def run_worker(engine: str, sbml: str, t_end: str, n_points: int, out: Path) -> int:
    """Worker entrypoint: run one engine, write the npz, never raise."""
    warnings.filterwarnings("ignore")
    try:
        res = _WORKERS[engine](sbml, t_end, n_points)
    except Exception as exc:  # defensive -- _phased should already trap this
        _save(
            out,
            load_ok=0,
            sim_ok=0,
            load_s=0.0,
            sim_s=0.0,
            t_end=0.0,
            error=f"{type(exc).__name__}: {str(exc)[:200]}",
        )
        return 0
    # finite-trajectory check -- a non-finite trajectory is not a real sim
    if res.get("sim_ok") and "data" in res and not np.all(np.isfinite(res["data"])):
        res["sim_ok"] = 0
        res["error"] = "non-finite trajectory"
    _save(out, **res)
    return 0


# ==========================================================================
# parent -- orchestrates the per-model, per-engine subprocess sweep
# ==========================================================================
class EngineResult:
    """Outcome of one engine subprocess on one model."""

    __slots__ = (
        "load_ok",
        "sim_ok",
        "wall_s",
        "load_s",
        "sim_s",
        "t_end",
        "is_steady",
        "error",
        "time",
        "names",
        "data",
    )

    def __init__(self):
        self.load_ok = self.sim_ok = False
        self.wall_s = self.load_s = self.sim_s = 0.0
        self.t_end = 0.0
        self.is_steady = False
        self.error = ""
        self.time = self.names = self.data = None


def run_engine(engine: str, sbml: Path, t_end, n_points: int, timeout: float) -> EngineResult:
    """Spawn one engine worker as an isolated, timeout-bounded subprocess."""
    er = EngineResult()
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tf:
        out = Path(tf.name)
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run.py"),
        "--worker",
        "--engine",
        engine,
        "--sbml",
        str(sbml),
        "--t-end",
        str(t_end),
        "--n-points",
        str(n_points),
        "--out",
        str(out),
    ]
    t0 = time.perf_counter()
    try:
        subprocess.run(cmd, timeout=timeout, capture_output=True)
    except subprocess.TimeoutExpired:
        er.wall_s = time.perf_counter() - t0
        er.error = f"timeout >{timeout:.0f}s"
        out.unlink(missing_ok=True)
        return er
    er.wall_s = time.perf_counter() - t0

    if not out.exists() or out.stat().st_size == 0:
        er.error = "worker produced no output (crash?)"
        out.unlink(missing_ok=True)
        return er
    try:
        with np.load(out, allow_pickle=True) as npz:
            er.error = str(npz["error"])
            er.load_s = float(npz["load_s"])
            er.sim_s = float(npz["sim_s"])
            er.t_end = float(npz["t_end"])
            er.load_ok = bool(int(npz["load_ok"]))
            er.sim_ok = bool(int(npz["sim_ok"]))
            if "data" in npz.files:
                er.is_steady = bool(int(npz["is_steady"]))
                er.time = npz["time"]
                er.names = [str(n) for n in npz["names"]]
                er.data = npz["data"]
    except Exception as exc:
        er.error = f"unreadable worker output: {exc}"
    finally:
        out.unlink(missing_ok=True)
    return er


def accuracy(ref: EngineResult, other: EngineResult) -> tuple[float | None, str]:
    """Max relative error of ``other`` vs the RR reference on shared species.

    Each species is scored in the max norm -- ``max_t|other-ref|`` --
    divided by ``max(own_peak, ACC_REL_FLOOR * model_scale)``, where
    ``own_peak`` is that species' own peak magnitude and ``model_scale``
    is the largest peak across all shared species.

    A species that carries real signal is divided by its own peak, so
    it is held to a strict per-species relative tolerance. A species
    that decays to the floating-point noise floor has an ``own_peak``
    far below the model scale; dividing its (meaningless) inter-engine
    noise by that tiny peak would manufacture a huge relative error, so
    the denominator is additionally floored at ``ACC_REL_FLOOR`` of the
    model scale and at the absolute ``ACC_ATOL`` -- a species pinned at
    ~1e-22 with a ~1e-14 inter-engine difference is then correctly
    scored as negligible rather than as a 1e-3 "error".

    Returns ``(max_rel_err, match_note)``. ``max_rel_err`` is ``None``
    when no comparison is possible (an engine failed, or no shared
    species / timepoints)."""
    if not (ref.sim_ok and other.sim_ok):
        return None, "engine failed"
    if ref.time is None or other.time is None:
        return None, "no trajectory"
    if len(ref.time) != len(other.time):
        return None, "timepoint count differs"

    ref_idx = {n: i for i, n in enumerate(ref.names)}
    shared = [(ref_idx[n], j) for j, n in enumerate(other.names) if n in ref_idx]
    if not shared:
        return None, "no shared species"

    peaks = [float(np.max(np.abs(ref.data[:, ri]))) for ri, _ in shared]
    model_scale = max(max(peaks), ACC_ATOL)
    floor = max(ACC_REL_FLOOR * model_scale, ACC_ATOL)

    max_err = 0.0
    for (ri, oi), peak in zip(shared, peaks, strict=False):
        abs_err = float(np.max(np.abs(other.data[:, oi] - ref.data[:, ri])))
        max_err = max(max_err, abs_err / max(peak, floor))
    return max_err, f"{len(shared)}/{len(other.names)} species"


# --- manifest --------------------------------------------------------------
def load_keep_models(manifest: Path) -> list[dict]:
    """Return the ``keep``-verdict manifest rows, with an effort tier added."""
    rows = []
    with open(manifest) as fh:
        # skip the leading "# ..." provenance comment line
        first = fh.readline()
        if not first.startswith("#"):
            fh.seek(0)
        for row in csv.DictReader(fh):
            if row["verdict"] != "keep":
                continue
            row["effort"] = size_tier(int(row["n_species"]), int(row["n_reactions"]))
            row["is_events"] = "events" in (row.get("tags") or "").split(";")
            rows.append(row)
    return rows


# --- per-model record ------------------------------------------------------
CSV_FIELDS = [
    "model_id",
    "events",
    "n_species",
    "n_reactions",
    "effort",
    "t_end",
    "is_steady",
    "rr_load_ok",
    "rr_sim_ok",
    "rr_wall_s",
    "rr_error",
    "bngsim_load_ok",
    "bngsim_sim_ok",
    "bngsim_wall_s",
    "bngsim_error",
    "amici_load_ok",
    "amici_sim_ok",
    "amici_wall_s",
    "amici_error",
    "bngsim_vs_rr_relerr",
    "bngsim_acc_ok",
    "bngsim_match",
    "amici_vs_rr_relerr",
    "amici_acc_ok",
    "amici_match",
]


# engine name -> CSV column prefix
ENGINE_PREFIX = {"roadrunner": "rr", "bngsim": "bngsim", "amici": "amici"}


def evaluate_model(
    row: dict, sbml_dir: Path, timeout: float, verbose: bool, engines: set[str]
) -> dict:
    """Run the selected engines on one model and assemble its record.

    ``engines`` is a subset of :data:`ENGINES`. Columns for engines not
    selected are left blank, so the CSV schema is stable across runs
    (e.g. a RoadRunner+BNGsim sweep now, AMICI filled in later)."""
    mid = row["model_id"]
    sbml = sbml_dir / f"{mid}.xml"
    rec = dict.fromkeys(CSV_FIELDS, "")
    rec.update(
        model_id=mid,
        events=int(row["is_events"]),
        n_species=row["n_species"],
        n_reactions=row["n_reactions"],
        effort=row["effort"],
    )

    if not sbml.exists():
        for eng in engines:
            rec[f"{ENGINE_PREFIX[eng]}_error"] = "sbml file missing"
        return rec

    results: dict[str, EngineResult] = {}

    # RoadRunner first when selected -- it fixes the horizon (and is the
    # accuracy reference) that the other engines reuse.
    rr = None
    if "roadrunner" in engines:
        rr = run_engine("roadrunner", sbml, "auto", N_POINTS, timeout)
        results["roadrunner"] = rr
        t_end = rr.t_end if rr.sim_ok else T_CANDIDATES[1]
        rec.update(
            is_steady=int(rr.is_steady),
            rr_load_ok=int(rr.load_ok),
            rr_sim_ok=int(rr.sim_ok),
            rr_wall_s=f"{rr.wall_s:.3f}",
            rr_error=rr.error,
        )
    else:
        # No reference available -- a fixed horizon, no accuracy column.
        t_end = T_FALLBACK
    rec["t_end"] = f"{t_end:g}"

    for eng in ("bngsim", "amici"):
        if eng not in engines:
            continue
        er = run_engine(eng, sbml, t_end, N_POINTS, timeout)
        results[eng] = er
        rec.update(
            {
                f"{eng}_load_ok": int(er.load_ok),
                f"{eng}_sim_ok": int(er.sim_ok),
                f"{eng}_wall_s": f"{er.wall_s:.3f}",
                f"{eng}_error": er.error,
            }
        )
        if rr is not None:
            err, note = accuracy(rr, er)
            rec[f"{eng}_match"] = note
            if err is not None:
                rec[f"{eng}_vs_rr_relerr"] = f"{err:.2e}"
                rec[f"{eng}_acc_ok"] = int(err <= ACC_RTOL)

    if verbose:
        for eng in ("roadrunner", "bngsim", "amici"):
            if eng not in results:
                continue
            acc = rec.get(f"{ENGINE_PREFIX[eng]}_vs_rr_relerr") or "-"
            print(f"    {eng:10s} {_fmt(results[eng])}  acc={acc}")
    return rec


def _fmt(er: EngineResult) -> str:
    if er.sim_ok:
        return f"OK  {er.wall_s:6.2f}s (load {er.load_s:.2f} sim {er.sim_s:.2f})"
    return f"FAIL  {er.wall_s:6.2f}s  {er.error}"


# --- output ----------------------------------------------------------------
def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(records)


def summarize(label: str, records: list[dict], engines: set[str]) -> None:
    n = len(records)
    if n == 0:
        print(f"  {label}: (no models)")
        return
    print(f"  {label}: {n} models")
    for eng in ("roadrunner", "bngsim", "amici"):
        if eng not in engines:
            continue
        pre = ENGINE_PREFIX[eng]
        loaded = sum(int(r[f"{pre}_load_ok"] or 0) for r in records)
        simmed = sum(int(r[f"{pre}_sim_ok"] or 0) for r in records)
        print(f"    {pre:8s} load {loaded:4d}/{n}   simulate {simmed:4d}/{n}")
    for eng in ("bngsim", "amici"):
        if eng not in engines or "roadrunner" not in engines:
            continue
        acc = sum(1 for r in records if r[f"{eng}_acc_ok"] == 1)
        cmp_ = sum(1 for r in records if r[f"{eng}_acc_ok"] != "")
        print(f"    {eng:8s} accuracy vs RR  {acc:4d}/{cmp_} within {ACC_RTOL:g}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage-2 BioModels suite runner")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--engine", choices=list(ENGINES))
    ap.add_argument("--sbml")
    ap.add_argument("--t-end", default="auto")
    ap.add_argument("--n-points", type=int, default=N_POINTS)
    ap.add_argument("--out")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--sbml-dir", type=Path, default=DEFAULT_SBML_DIR)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    ap.add_argument("--model", help="run a single model by id")
    ap.add_argument("--limit", type=int, default=None, help="cap on models")
    ap.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT, help="per-engine wall-clock budget (s)"
    )
    ap.add_argument(
        "--engines",
        default=",".join(ENGINES),
        help="comma-separated engines to run (default: all). e.g. "
        "'roadrunner,bngsim' skips the slow AMICI per-model compile.",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    add_effort_arg(ap)
    args = ap.parse_args()

    # ---- worker mode -----------------------------------------------------
    if args.worker:
        return run_worker(args.engine, args.sbml, args.t_end, args.n_points, Path(args.out))

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    bad = [e for e in engines if e not in ENGINES]
    if bad:
        print(
            f"unknown engine(s): {', '.join(bad)} (choose from {', '.join(ENGINES)})",
            file=sys.stderr,
        )
        return 1
    engines = set(engines)

    # ---- parent mode -----------------------------------------------------
    models = load_keep_models(args.manifest)
    if args.model:
        models = [m for m in models if m["model_id"] == args.model]
    else:
        models = filter_by_effort(models, args.effort, key=lambda m: m["effort"])
    if args.limit:
        models = models[: args.limit]
    if not models:
        print("no models selected", file=sys.stderr)
        return 1

    eng_order = [e for e in ENGINES if e in engines]
    print(
        f"BioModels Stage-2 runner | {len(models)} models "
        f"| effort={args.effort} | engines={'+'.join(eng_order)} "
        f"| per-engine timeout={args.timeout:g}s"
    )
    print("=" * 70)

    records = []
    t_start = time.perf_counter()
    for i, row in enumerate(models, 1):
        print(
            f"[{i}/{len(models)}] {row['model_id']}"
            f" ({row['n_species']}sp/{row['n_reactions']}rxn,"
            f" {row['effort']}{', events' if row['is_events'] else ''})",
            flush=True,
        )
        records.append(evaluate_model(row, args.sbml_dir, args.timeout, args.verbose, engines))
    wall = time.perf_counter() - t_start

    events = [r for r in records if r["events"] == 1]
    non_events = [r for r in records if r["events"] == 0]

    args.results_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.results_dir / "coverage.csv", records)
    write_csv(args.results_dir / "figS1_biomodels.csv", non_events)
    write_csv(args.results_dir / "figS2_events.csv", events)

    print("\n" + "=" * 70)
    print(f"SUMMARY  ({len(records)} models, {wall:.1f}s wall)")
    summarize("Fig S1  non-events", non_events, engines)
    summarize("Fig S2  events    ", events, engines)
    print(f"\n  coverage.csv          -> {args.results_dir / 'coverage.csv'}")
    print(f"  figS1_biomodels.csv   -> {args.results_dir / 'figS1_biomodels.csv'}")
    print(f"  figS2_events.csv      -> {args.results_dir / 'figS2_events.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
