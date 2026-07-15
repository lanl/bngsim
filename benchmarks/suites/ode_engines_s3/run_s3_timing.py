#!/usr/bin/env python3
"""Table S3 — cross-engine ODE timing on the six representative published BNGL models.

For each of the six Table-S1 models (Lang_2024, Kocieniewski_2012, Barua_2007,
Blinov_2006, Barua_2013, fceri_fyn) this times an ODE integration under six engine
configurations:

    run_network · BNGsim · RoadRunner · AMICI-KLU · AMICI-dense · COPASI

Metric (agreed with Bill 2026-07-15):
  * COLD cost  = one-time model build/setup + the FIRST integration
                 (``cold_total_sec`` = ``build_sec`` + ``integrate_cold_sec``).
  * WARM cost  = the median of the SUBSEQUENT replicate integrations
                 (``warm_sec`` = ``integrate_warm_median_sec``); the number of warm
                 reps is the shared adaptive budget (``_warm_rep_count``:
                 WARM_REPS=5 capped at WARM_BUDGET_SEC=3 s of wall).

``build_sec`` means the one-time cost each engine pays before it can integrate:
  * run_network — 0 (every call is a fresh CVODE process; the read+setup lands in
                  ``integrate_cold_sec`` and again in each warm call).
  * BNGsim      — ``from_net`` C++ load (the lazy analytic-Jacobian derivation and
                  RHS codegen fold into the first ``sim.run`` = ``integrate_cold_sec``).
  * RoadRunner  — ``RoadRunner(sbml)`` construction (libSBML parse + LLVM JIT).
  * AMICI       — the C++ codegen + compile + extension load (the ~15-30 s floor);
                  the SAME compiled model serves BOTH the KLU and dense rows, so
                  ``build_sec`` is repeated in both (it is paid once — see the note
                  in the report).
  * COPASI      — ``importSBML`` + Time-Course task configuration.

SBML-only engines (RoadRunner / AMICI / COPASI) consume the **SBML + SED-ML that
BNGsim's OMEX export produces** (``convert.net_to_omex``, with ``bngl=`` so the
bundled SED-ML carries the model's own protocol). Every engine integrates the SAME
model over the SAME horizon at the SAME tolerances, so the per-engine cost is a fair
comparison.

DATA SOURCES (read-only): the six models' FULL ``.net`` networks are the cache built
by the ``ode_fullnet`` benchmark (Phase 1); the horizon is each model's own ODE
``simulate`` (``bc.parse_ode_spec``) with a documented default fallback.

ENVIRONMENT: run under the only venv that carries all six engines —
``~/Code/PyBNF-Private/bngsim/.venv`` (bngsim / roadrunner / amici / COPASI). Set
``BNGPATH`` to a BioNetGen-2.9.3 tree for ``run_network``. Run SERIAL for published
numbers (timing must not contend). Each engine call is fork-isolated with a wall
timeout, so a crash or hang on the giant (fceri_fyn) is contained, not fatal.

    export BNGPATH=~/Simulations/BioNetGen-2.9.3
    ~/Code/PyBNF-Private/bngsim/.venv/bin/python run_s3_timing.py --fresh-amici
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import select
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Resolve the parity/benchmark modules we reuse. BNGSIM_ROOT points at the tree
# that holds parity_checks + the ode_fullnet .net cache (the CANONICAL bngsim
# checkout ~/Code/bngsim), independent of which venv/bngsim actually runs.
# --------------------------------------------------------------------------- #
BNGSIM = Path(os.environ.get("BNGSIM_ROOT", Path.home() / "Code" / "bngsim"))
PARITY = BNGSIM / "parity_checks"
BNG_PARITY = PARITY / "bng_parity"
AMICI_PARITY = PARITY / "amici_parity"
FULLNET = BNGSIM / "benchmarks" / "suites" / "ode_fullnet"
for _p in (str(PARITY), str(BNG_PARITY), str(AMICI_PARITY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bng_common as bc  # noqa: E402
from _core import read_manifest  # noqa: E402
from rr_parity import _rr_common as rc  # noqa: E402

HERE = Path(__file__).resolve().parent
OMEX_DIR = HERE / "omex"
SBML_DIR = HERE / "sbml"
OUT = HERE / "report_ode_engines_s3.json"
JOBS = BNG_PARITY / "jobs.json"
MODELS_ROOT = BNG_PARITY / "models"
NETS_MANIFEST = FULLNET / "nets_manifest.json"
NETS_DIR = FULLNET / "nets"

# The six Table-S1 models, in table order 1..6. Each is matched against the
# ode_fullnet manifest by the path fragment; ``label`` is the display name and
# ``row`` the Fig-S4 / Table-S1 numbering.
MODELS = [
    {"label": "Lang_2024", "key": "/Lang2024/", "row": 1},
    {"label": "Kocieniewski_2012", "key": "/Kocieniewski2012/", "row": 2},
    {"label": "Barua_2007", "key": "/Barua2007/", "row": 3},
    {"label": "Blinov_2006", "key": "/Blinov2006/", "row": 4},
    {"label": "Barua_2013", "key": "/Barua2013/", "row": 5},
    {"label": "fceri_fyn", "key": "/fcerifyn/", "row": 6},
]

# --------------------------------------------------------------------------- #
# AUTHOR HORIZON OVERRIDES.
# Kocieniewski_2012 and fceri_fyn ship NO runnable ODE `simulate` action in their
# RuleHub BNGL (verified against .sources_cache 2026-07-15: neither the parity
# model nor the original source carries one), so both currently fall back to the
# DEFAULT horizon below (flagged horizon_source="default" in the report). Bill is
# supplying the author-specified time horizons from the publications; drop them in
# here (keyed by the MODELS `key` fragment) and re-run:
#     --redo --models Kocieniewski2012,fcerifyn
# Values are (t_end, n_steps) with t_start=0; add t_start via a 3-tuple/dict if a
# model needs it.
# --------------------------------------------------------------------------- #
HORIZON_OVERRIDES: dict[str, dict] = {
    # "/Kocieniewski2012/": {"t_end": ____, "n_steps": ____},   # TODO: from Kocieniewski et al. 2012
    # "/fcerifyn/":        {"t_end": ____, "n_steps": ____},   # TODO: from the BNG FcERI/Fyn example
}

DEFAULT_TEND = 100.0
DEFAULT_NSTEPS = 100

# Engine columns in report order.
ENGINE_ORDER = ["run_network", "bngsim", "roadrunner", "amici_klu", "amici_dense", "copasi"]


# --------------------------------------------------------------------------- #
# Normalized per-engine result.
# --------------------------------------------------------------------------- #
def _norm(build_sec, build_breakdown, cached, integ, config, n_species, extra=None):
    """Assemble the uniform per-engine timing dict from a build cost + an
    ``_integrate_stats`` result. ``cold_total_sec`` = build + first solve is the
    reported COLD cost; ``warm_sec`` = warm median is the reported WARM cost."""
    cold_int = integ.get("integrate_cold_sec")
    d = {
        "status": "ok",
        "n_species": n_species,
        "build_sec": None if build_sec is None else round(build_sec, 6),
        "build_breakdown": build_breakdown,
        "build_cached": cached,
        "integrate_cold_sec": cold_int,
        "integrate_warm_min_sec": integ.get("integrate_warm_min_sec"),
        "integrate_warm_median_sec": integ.get("integrate_warm_median_sec"),
        "integrate_warm_max_sec": integ.get("integrate_warm_max_sec"),
        "n_warm": integ.get("integrate_n_warm"),
        "cold_total_sec": (
            round((build_sec or 0.0) + cold_int, 6) if cold_int is not None else None
        ),
        "warm_sec": integ.get("integrate_warm_median_sec"),
        "config": config,
    }
    if extra:
        d.update(extra)
    return d


def _err(msg, status="error"):
    return {"status": status, "error": str(msg)[:600]}


# --------------------------------------------------------------------------- #
# Engine adapters. Each runs in a forked child (see run_in_child) and returns a
# JSON-serializable dict. bngsim / run_network / RoadRunner reuse the canonical
# parity adapters; AMICI and COPASI are local (AMICI to time KLU+dense off one
# compile; COPASI has no existing timing adapter).
# --------------------------------------------------------------------------- #
def eng_bngsim(net_path, hz):
    t, v, names, tm = bc.bn_ode_net(
        net_path, hz["t_start"], hz["t_end"], hz["n_points"], hz["rtol"], hz["atol"]
    )
    if not np.isfinite(v).all():
        return _err("bngsim produced a non-finite trajectory")
    n = int(v.shape[1])
    # from_net C++ load is the build; the lazy Jacobian derivation + RHS codegen
    # happen INSIDE the first sim.run, so they are already inside integrate_cold_sec
    # (reported in build_breakdown for transparency, NOT added to build_sec).
    build = float(tm.get("load_sec") or 0.0)
    bd = {k: tm.get(k) for k in ("load_sec", "jac_derive_sec", "codegen_sec")}
    cfg = tm.get("config", {})
    return _norm(build, bd, cfg.get("cached"), tm, cfg, n)


def eng_run_network(net_path, hz, rn_bin):
    work = Path(tempfile.mkdtemp(prefix="s3rn_"))
    try:
        t, v, names, tm = bc.run_network_ode(
            net_path, rn_bin,
            t_start=hz["t_start"], t_end=hz["t_end"], n_steps=hz["n_steps"],
            rtol=hz["rtol"], atol=hz["atol"], out_prefix=str(work / "rn"),
            timeout=hz["rn_timeout"],
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if not np.isfinite(v).all():
        return _err("run_network produced a non-finite trajectory")
    n = int(v.shape[1])
    # No persistent build: each call is a fresh process, so the read+CVODE-setup
    # cost lands inside integrate_cold_sec (first call) and every warm call.
    bd = {k: tm.get(k) for k in ("init_cpu_sec", "propagation_cpu_sec", "total_cpu_sec")}
    return _norm(0.0, bd, None, tm, tm.get("config", {}), n,
                 extra={"n_calls": tm.get("n_calls")})


def eng_roadrunner(sbml_path, hz):
    t, v, names, tm = rc.rr_ode(
        str(sbml_path), hz["t_start"], hz["t_end"], hz["n_points"], hz["rtol"], hz["atol"]
    )
    if not np.isfinite(v).all():
        return _err("roadrunner produced a non-finite trajectory")
    n = int(v.shape[1])
    # RoadRunner(sbml) construction = libSBML parse + LLVM JIT (the one-time build).
    build = float(tm.get("parse_interpret_codegen_sec") or 0.0)
    bd = {k: tm.get(k) for k in
          ("read_sec", "parse_sec", "interpret_sec", "codegen_sec", "jit_sec",
           "parse_interpret_codegen_sec", "model_cache_hit")}
    cfg = tm.get("config", {})
    return _norm(build, bd, cfg.get("cached"), tm, cfg, n)


def eng_amici(sbml_path, hz, fresh):
    """Time BOTH AMICI linear-solver variants off a SINGLE compile. Returns a dict
    with keys ``amici_klu`` and ``amici_dense`` (plus ``_note``). The C++ compile is
    the shared one-time build recorded in both rows' ``build_sec`` (paid once)."""
    import amici  # noqa: F401
    import amici.sim.sundials as ss
    import _amici_common as amc

    sbml_str = Path(sbml_path).read_text()
    if fresh:
        key = hashlib.sha256((sbml_str + amc._BUILD_FLAGS).encode()).hexdigest()[:16]
        shutil.rmtree(amc.AMICI_CACHE / f"amici_{key}", ignore_errors=True)

    model, bt, cached = amc._build_model(sbml_str)
    build = (bt["parse_sec"] + bt["interpret_sec"] + bt["jac_derive_sec"]
             + bt["codegen_sec"] + bt["compile_sec"] + bt["load_sec"])
    ts = np.linspace(hz["t_start"], hz["t_end"], hz["n_points"])
    sens_none = getattr(ss, "SensitivityOrder_none", 0)

    out = {"_note": ("AMICI codegen+compile is one-time and shared by amici_klu and "
                     "amici_dense (paid once); build_sec repeats it in both rows.")}
    for eng, enumval in (("amici_klu", ss.LinearSolver_KLU),
                         ("amici_dense", ss.LinearSolver_dense)):
        try:
            solver = model.create_solver()
            solver.set_relative_tolerance(hz["rtol"])
            solver.set_absolute_tolerance(hz["atol"])
            solver.set_sensitivity_order(sens_none)
            solver.set_linear_solver(enumval)
            model.set_timepoints(ts)

            t1 = time.perf_counter()
            rd = model.simulate(solver=solver)
            cold = time.perf_counter() - t1
            if int(rd.status) != 0:
                out[eng] = {**_err(f"AMICI status {rd.status} (cold)", "error"),
                            "build_sec": round(build, 6), "build_breakdown": bt}
                continue
            warm = []
            for _ in range(rc._warm_rep_count(cold)):
                try:
                    t1 = time.perf_counter()
                    r2 = model.simulate(solver=solver)
                    if int(r2.status) != 0:
                        break
                    warm.append(time.perf_counter() - t1)
                except Exception:
                    break
            integ = rc._integrate_stats(cold, warm)
            try:
                ls = ss.LinearSolver(solver.get_linear_solver()).name
            except Exception:
                ls = str(solver.get_linear_solver())
            n = int(np.asarray(rd.x).shape[1])
            cfg = {"codegen": "C++ (compiled)", "jacobian": "analytical (symbolic)",
                   "linear_solver": ls, "cached": cached}
            out[eng] = _norm(build, bt, cached, integ, cfg, n,
                             extra={"integrate_cpu_ms": round(float(rd.cpu_time), 4)})
        except Exception as e:  # noqa: BLE001
            out[eng] = {**_err(f"{type(e).__name__}: {e}"),
                        "build_sec": round(build, 6), "build_breakdown": bt}
    return out


def eng_copasi(sbml_path, hz):
    import COPASI

    dm = COPASI.CRootContainer.addDatamodel()
    try:
        t0 = time.perf_counter()
        try:
            if not dm.importSBML(str(sbml_path)):
                return _err("COPASI importSBML returned False", "load_fail")
        except Exception as e:  # noqa: BLE001
            return _err(f"COPASI importSBML: {e}", "load_fail")
        task = dm.getTask("Time-Course")
        task.setMethodType(COPASI.CTaskEnum.Method_deterministic)
        task.setScheduled(True)
        prob = task.getProblem()
        prob.setDuration(hz["t_end"] - hz["t_start"])
        prob.setStepNumber(hz["n_steps"])
        prob.setOutputStartTime(hz["t_start"])
        prob.setTimeSeriesRequested(True)
        mth = task.getMethod()
        for pn, pv in (("Absolute Tolerance", hz["atol"]), ("Relative Tolerance", hz["rtol"])):
            p = mth.getParameter(pn)
            if p is not None:
                p.setValue(pv)
        build = time.perf_counter() - t0
        n = int(dm.getModel().getNumMetabs())

        def _one():
            if not task.initialize(COPASI.CCopasiTask.OUTPUT_UI):
                raise RuntimeError("COPASI task init failed")
            if not task.process(True):
                raise RuntimeError("COPASI task process failed")

        t1 = time.perf_counter()
        _one()
        cold = time.perf_counter() - t1
        warm = []
        for _ in range(rc._warm_rep_count(cold)):
            try:
                t1 = time.perf_counter()
                _one()
                warm.append(time.perf_counter() - t1)
            except Exception:
                break
        integ = rc._integrate_stats(cold, warm)
        cfg = {"codegen": "interpreted (COPASI)", "jacobian": "internal",
               "linear_solver": "COPASI deterministic (LSODA)", "cached": None}
        return _norm(build, {"import_config_sec": round(build, 6)}, None, integ, cfg, n)
    finally:
        with contextlib.suppress(Exception):
            COPASI.CRootContainer.removeDatamodel(dm)


# --------------------------------------------------------------------------- #
# Fork isolation + wall timeout. Each engine call runs in a child so a native
# crash (COPASI segfaults on some models; a giant AMICI/RR JIT may OOM) or a hang
# is contained and reported, never fatal to the sweep. Timing is measured INSIDE
# the child by the adapters, so the fork/pipe overhead is not charged.
# --------------------------------------------------------------------------- #
def run_in_child(fn, timeout):
    if not hasattr(os, "fork"):  # pragma: no cover — non-POSIX fallback
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            return _err(f"{type(e).__name__}: {e}")
    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(r_fd)
        try:
            res = fn()
        except Exception as e:  # noqa: BLE001
            res = _err(f"{type(e).__name__}: {e}")
        try:
            os.write(w_fd, json.dumps(res).encode())
        except Exception:
            with contextlib.suppress(Exception):
                os.write(w_fd, json.dumps(_err("child produced non-serializable result")).encode())
        finally:
            os.close(w_fd)
            os._exit(0)
    # parent
    os.close(w_fd)
    chunks: list[bytes] = []
    deadline = time.monotonic() + timeout
    timed_out = False
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        rlist, _, _ = select.select([r_fd], [], [], remaining)
        if not rlist:
            timed_out = True
            break
        b = os.read(r_fd, 65536)
        if not b:
            break
        chunks.append(b)
    os.close(r_fd)
    if timed_out:
        with contextlib.suppress(Exception):
            os.kill(pid, 9)
        with contextlib.suppress(Exception):
            os.waitpid(pid, 0)
        return _err(f"engine exceeded wall timeout of {timeout:.0f}s", "timeout")
    _, status = os.waitpid(pid, 0)
    if os.WIFSIGNALED(status):
        return _err(f"engine crashed in native code (signal {os.WTERMSIG(status)})", "error")
    try:
        return json.loads(b"".join(chunks))
    except Exception:
        return _err("engine produced no parseable result")


# --------------------------------------------------------------------------- #
# OMEX export (SBML + SED-ML) via bngsim's own converter. Cached per model.
# --------------------------------------------------------------------------- #
def ensure_sbml(net_path, bngl_path, san, gate, hz):
    """Export the .net to an OMEX archive (SBML + SED-ML) and extract model.xml.
    Cached: returns the extracted SBML path + export seconds. Runs in the caller's
    process (bngsim); wrap the call in run_in_child at the call site for a timeout."""
    from bngsim.convert import net_to_omex, read_omex

    OMEX_DIR.mkdir(parents=True, exist_ok=True)
    SBML_DIR.mkdir(parents=True, exist_ok=True)
    omex_path = OMEX_DIR / f"{san}.omex"
    sbml_out = SBML_DIR / f"{san}.xml"
    if sbml_out.exists() and omex_path.exists():
        return {"status": "ok", "sbml_path": str(sbml_out), "omex_path": str(omex_path),
                "export_sec": None, "cached": True}
    t0 = time.perf_counter()
    rep = net_to_omex(str(net_path), str(omex_path), bngl=str(bngl_path), gate=gate)
    export_sec = time.perf_counter() - t0
    ex = SBML_DIR / san
    arch = read_omex(str(omex_path), extract_dir=str(ex))
    master = arch.master_model_entry()
    src = arch.path_of(master)
    shutil.copyfile(src, sbml_out)
    return {"status": "ok", "sbml_path": str(sbml_out), "omex_path": str(omex_path),
            "export_sec": round(export_sec, 4), "cached": False, "gate_ok": bool(rep.ok)}


# --------------------------------------------------------------------------- #
def resolve_run_network() -> str:
    if os.environ.get("RUN_NETWORK"):
        p = Path(os.environ["RUN_NETWORK"])
    else:
        root = Path(os.environ.get("BNGPATH", str(Path.home() / "Simulations" / "BioNetGen-2.9.3")))
        root = root.parent if root.name == "BNG2.pl" else root
        p = root / "bin" / "run_network"
    if not p.exists():
        sys.exit(f"ABORT: run_network not found at {p} (set BNGPATH to a BioNetGen-2.9.3 root)")
    return str(p)


def _versions() -> dict:
    v = {}
    for mod in ("bngsim", "roadrunner", "amici", "libsbml"):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            v[mod] = f"unavailable ({type(e).__name__})"
    try:
        import COPASI
        v["copasi"] = COPASI.CVersion.VERSION.getVersion() if hasattr(COPASI, "CVersion") else "?"
    except Exception as e:  # noqa: BLE001
        v["copasi"] = f"unavailable ({type(e).__name__})"
    with contextlib.suppress(Exception):
        v["sundials"] = bc.sundials_version()
    return v


def resolve_models(args):
    nets = json.loads(NETS_MANIFEST.read_text())
    _meta, alljobs = read_manifest(JOBS)
    rel = {j.model_id: j.model for j in alljobs}
    picked = []
    for spec in MODELS:
        mid = next((m for m in nets
                    if spec["key"] in m and nets[m].get("status") == "ok"), None)
        if mid is None:
            print(f"  WARN: no cached net for {spec['label']} ({spec['key']}) — skipped")
            continue
        picked.append({**spec, "model_id": mid, "net_file": nets[mid]["net_file"],
                       "n_species": nets[mid].get("n_species"),
                       "n_reactions": nets[mid].get("n_reactions"),
                       "model_rel": rel.get(mid, mid)})
    if args.models:
        subs = [s.strip() for s in args.models.split(",") if s.strip()]
        picked = [p for p in picked if any(s in p["model_id"] or s in p["label"] for s in subs)]
    if args.limit:
        picked = picked[: args.limit]
    return picked


def horizon_for(spec, args):
    net_path = NETS_DIR / spec["net_file"]
    bngl_text = (MODELS_ROOT / spec["model_rel"]).read_text(errors="replace")
    ov = HORIZON_OVERRIDES.get(spec["key"])
    if ov:
        t_start = float(ov.get("t_start", 0.0))
        t_end = float(ov["t_end"])
        n_steps = int(ov["n_steps"])
        source = "override"
    else:
        net_params = bc.read_net_parameters(net_path)
        ode = bc.parse_ode_spec(bngl_text, net_params, atol=args.atol, rtol=args.rtol)
        if ode is None:
            t_start, t_end, n_steps, source = 0.0, args.default_tend, args.default_nsteps, "default"
        else:
            t_start, t_end, n_steps, source = (
                float(ode["t_start"]), float(ode["t_end"]), int(ode["n_steps"]), "model")
    return {
        "t_start": t_start, "t_end": t_end, "n_steps": n_steps, "n_points": n_steps + 1,
        "rtol": args.rtol, "atol": args.atol, "rn_timeout": args.timeout,
        "horizon_source": source,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="", help="Comma-separated model_id/label substring filter.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--redo", action="store_true", help="Re-time models already present in the report.")
    ap.add_argument("--fresh-amici", action="store_true",
                    help="Clear each model's AMICI compile cache first so build_sec is a true cold compile.")
    ap.add_argument("--omex-gate", default="full", choices=["full", "L1", "none"],
                    help="net->SBML validation gate for the OMEX export (default full).")
    ap.add_argument("--rtol", type=float, default=bc.DEFAULT_RTOL)
    ap.add_argument("--atol", type=float, default=bc.DEFAULT_ATOL)
    ap.add_argument("--default-tend", type=float, default=DEFAULT_TEND)
    ap.add_argument("--default-nsteps", type=int, default=DEFAULT_NSTEPS)
    ap.add_argument("--timeout", type=float, default=1200.0,
                    help="Per-engine wall timeout (s); also the run_network subprocess timeout.")
    ap.add_argument("--omex-timeout", type=float, default=1200.0)
    ap.add_argument("--engines", default=",".join(ENGINE_ORDER),
                    help="Comma-separated subset of engines to run.")
    args = ap.parse_args()

    rn_bin = resolve_run_network()
    want_engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    models = resolve_models(args)

    prior = {}
    if OUT.exists():
        with contextlib.suppress(Exception):
            prior = {r["model_id"]: r for r in json.loads(OUT.read_text()).get("results", [])}

    print("=" * 74)
    print("  Table S3 — cross-engine ODE timing (6 published models x 6 engines)")
    print("=" * 74)
    print(f"  models: {len(models)}   engines: {want_engines}")
    print(f"  run_network: {rn_bin}")
    print(f"  fresh-amici: {args.fresh_amici}   omex-gate: {args.omex_gate}   "
          f"tol: {args.rtol:g}/{args.atol:g}   timeout: {args.timeout:g}s")
    print(f"  bngsim: {_versions().get('bngsim')}")
    print()

    results = dict(prior)
    for spec in models:
        mid = spec["model_id"]
        if not args.redo and mid in prior and prior[mid].get("_complete"):
            print(f"  [skip] {spec['label']:20} (already complete)")
            continue
        san = mid.replace("/", "__")
        net_path = NETS_DIR / spec["net_file"]
        bngl_path = MODELS_ROOT / spec["model_rel"]
        hz = horizon_for(spec, args)
        print(f"  #{spec['row']} {spec['label']:20} N={spec['n_species']:>5} "
              f"rxn={spec['n_reactions']:>6}  horizon t_end={hz['t_end']:g} "
              f"n_steps={hz['n_steps']} ({hz['horizon_source']})")

        # OMEX export (SBML + SED-ML) — needed by the SBML engines. Fork-isolated.
        need_sbml = any(e in want_engines for e in ("roadrunner", "amici_klu", "amici_dense", "copasi"))
        sbml_path = None
        omex_info = None
        if need_sbml:
            omex_info = run_in_child(
                lambda: ensure_sbml(net_path, bngl_path, san, args.omex_gate, hz),
                args.omex_timeout)
            if omex_info.get("status") == "ok":
                sbml_path = omex_info["sbml_path"]
                tag = "cached" if omex_info.get("cached") else f"{omex_info.get('export_sec')}s"
                print(f"       omex/sbml: {tag}  gate_ok={omex_info.get('gate_ok')}")
            else:
                print(f"       omex/sbml: FAILED ({omex_info.get('error')})")

        engines: dict[str, dict] = {}
        for eng in want_engines:
            if eng == "run_network":
                res = run_in_child(lambda: eng_run_network(net_path, hz, rn_bin), args.timeout)
            elif eng == "bngsim":
                res = run_in_child(lambda: eng_bngsim(net_path, hz), args.timeout)
            elif eng in ("roadrunner", "amici_klu", "amici_dense", "copasi"):
                if sbml_path is None:
                    res = _err(f"no SBML (omex export failed): {(omex_info or {}).get('error')}", "skipped")
                elif eng == "roadrunner":
                    res = run_in_child(lambda: eng_roadrunner(sbml_path, hz), args.timeout)
                elif eng == "copasi":
                    res = run_in_child(lambda: eng_copasi(sbml_path, hz), args.timeout)
                else:  # amici_klu / amici_dense — computed together, once
                    if "amici_klu" in engines or "amici_dense" in engines:
                        res = engines.get(eng)  # already produced below
                    else:
                        amici_out = run_in_child(
                            lambda: eng_amici(sbml_path, hz, args.fresh_amici), args.timeout)
                        note = amici_out.get("_note") if isinstance(amici_out, dict) else None
                        for k in ("amici_klu", "amici_dense"):
                            sub = amici_out.get(k) if isinstance(amici_out, dict) else None
                            engines[k] = sub if sub is not None else dict(
                                amici_out if isinstance(amici_out, dict) else _err("amici child failed"))
                            if note and isinstance(engines[k], dict):
                                engines[k].setdefault("_note", note)
                        res = engines.get(eng)
            else:
                res = _err(f"unknown engine {eng}", "skipped")
            if res is not None:
                engines[eng] = res
            _print_engine(eng, engines.get(eng))

        results[mid] = {
            "model_id": mid, "label": spec["label"], "row": spec["row"],
            "n_species": spec["n_species"], "n_reactions": spec["n_reactions"],
            "horizon": hz, "omex": omex_info, "engines": engines,
            "_complete": all(e in engines for e in want_engines),
        }
        _write(results, args, want_engines)
        print()

    print(f"  report: {OUT}")
    return 0


def _fmt(x, ms=False):
    if not isinstance(x, (int, float)):
        return "   --   "
    return f"{x*1000:8.2f}ms" if ms else f"{x:8.3f}s"


def _print_engine(eng, r):
    if not r:
        print(f"       {eng:14} (none)")
        return
    if r.get("status") != "ok":
        print(f"       {eng:14} {r.get('status').upper()}: {(r.get('error') or '')[:60]}")
        return
    ls = (r.get("config") or {}).get("linear_solver", "")
    print(f"       {eng:14} cold={_fmt(r.get('cold_total_sec'))} "
          f"(build={_fmt(r.get('build_sec'))}+first={_fmt(r.get('integrate_cold_sec'))}) "
          f"warm={_fmt(r.get('warm_sec'), ms=True)}  {ls}")


def _write(results, args, want_engines):
    ordered = sorted(results.values(), key=lambda r: (r.get("n_species") or 0))
    doc = {
        "_meta": {
            "suite": "ode_engines_s3",
            "description": ("Cross-engine ODE timing on the six representative published "
                            "BNGL models (Table S1). Cold = one-time build + first solve; "
                            "warm = median of subsequent solves. SBML engines consume "
                            "bngsim net_to_omex SBML+SED-ML."),
            "engines": want_engines,
            "metric": {"cold": "build_sec + integrate_cold_sec (= cold_total_sec)",
                       "warm": "integrate_warm_median_sec (= warm_sec)",
                       "warm_reps": {"max": rc.WARM_REPS, "budget_sec": rc.WARM_BUDGET_SEC}},
            "rtol": args.rtol, "atol": args.atol,
            "default_horizon": {"t_end": args.default_tend, "n_steps": args.default_nsteps},
            "omex_gate": args.omex_gate,
            "versions": _versions(),
            "hardware": rc.hardware_info(),
            "n_results": len(ordered),
        },
        "results": ordered,
    }
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(OUT)


if __name__ == "__main__":
    raise SystemExit(main())
