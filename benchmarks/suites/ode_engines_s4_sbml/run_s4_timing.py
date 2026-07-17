#!/usr/bin/env python3
"""Table S4 — cross-engine ODE timing on the six largest curated SBML BioModels.

The SBML counterpart to Table S3 (``ode_engines_s3/run_s3_timing.py``). Where S3
starts from a BNGL/``.net`` reaction network and *derives* the SBML the SBML-only
engines consume, S4 is INVERTED: the source is a curated SBML BioModel and its
figure-reproducing SED-ML, and it is the ``.net`` for ``run_network`` that is
*derived* (via ``bngsim.sbml_to_net``).

The six models are the six largest (by bngsim species count ``N``) curated
BioModels that carry a real, figure-tied simulation horizon (a ``figure_sedml`` in
the rr_parity corpus):

    (1) BIOMD0000000834 Verma2016        N=76   (2) BIOMD0000000468 Koo2013        N=79
    (3) BIOMD0000000559 Ouzounoglou2014  N=90   (4) BIOMD0000000667 Hornberg2005   N=103
    (5) BIOMD0000000474 Smith2013        N=133  (6) BIOMD0000001065 vonDassow2000   N=148

Each is integrated over its own published-figure horizon (from ``ode_jobs.json``,
the ``figure_sedml`` tier) under six engine configurations:

    run_network · BNGsim · RoadRunner · AMICI-KLU · AMICI-dense · COPASI

Metric (identical to S3, agreed with Bill 2026-07-15):
  * COLD cost  = one-time model build/setup + the FIRST integration
                 (``cold_total_sec`` = ``build_sec`` + ``integrate_cold_sec``).
  * WARM cost  = median of the SUBSEQUENT replicate integrations
                 (``warm_sec`` = ``integrate_warm_median_sec``); the number of warm
                 reps is the shared adaptive budget (``_warm_rep_count``:
                 WARM_REPS=5 capped at WARM_BUDGET_SEC=3 s of wall).

Ingestion per engine:
  * RoadRunner / AMICI-KLU / AMICI-dense / COPASI — the curated SBML ``.xml``
    directly (native source), integrated over the SED-ML figure horizon.
  * BNGsim      — ``Model.from_sbml(xml)``.
  * run_network — a ``.net`` produced offline by ``bngsim.sbml_to_net`` (validate
    ``"L2"``, ``strict=True``, ``sedml=`` to carry the figure horizon), then the
    legacy CVODE binary. The SBML→``.net`` conversion is model-preparation, NOT
    charged to the engine (mirroring how S3's BNG2.pl network generation and the
    net→SBML OMEX export are model-preparation there); its cost + faithfulness are
    recorded in the per-model ``net_conversion`` field. Some SBML constructs the
    flat ``.net`` cannot carry faithfully make ``sbml_to_net`` (strict) refuse —
    e.g. BIOMD0000000474's four ``event`` blocks — leaving the run_network cell
    EMPTY (an honest "run_network can't do everything" finding).

``build_sec`` — the one-time cost each engine pays before it can integrate — has
the same meaning as in S3:
  * run_network — 0 (every call is a fresh CVODE process; read+setup lands in
                  ``integrate_cold_sec`` and again in each warm call).
  * BNGsim      — ``from_sbml`` load (libSBML parse + network build; the lazy
                  analytic-Jacobian derivation and RHS codegen fold into the first
                  ``sim.run`` = ``integrate_cold_sec``).
  * RoadRunner  — ``RoadRunner(sbml)`` construction (libSBML parse + LLVM JIT).
  * AMICI       — C++ codegen + compile + extension load; the SAME compiled model
                  serves BOTH the KLU and dense rows (paid once — see the report note).
  * COPASI      — ``importSBML`` + Time-Course task configuration.

ENVIRONMENT: run under the only venv that carries all six engines —
``~/Code/PyBNF-Private/bngsim/.venv`` (bngsim / roadrunner / amici / COPASI). Set
``BNGPATH`` to a BioNetGen-2.9.3 tree for ``run_network``. Run SERIAL for published
numbers. Each engine call is fork-isolated with a wall timeout.

    export BNGPATH=~/Simulations/BioNetGen-2.9.3
    ~/Code/PyBNF-Private/bngsim/.venv/bin/python run_s4_timing.py --fresh-amici
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
# that holds parity_checks (the CANONICAL bngsim checkout ~/Code/bngsim),
# independent of which venv/bngsim actually runs.
# --------------------------------------------------------------------------- #
BNGSIM = Path(os.environ.get("BNGSIM_ROOT", Path.home() / "Code" / "bngsim"))
PARITY = BNGSIM / "parity_checks"
BNG_PARITY = PARITY / "bng_parity"
AMICI_PARITY = PARITY / "amici_parity"
RR_PARITY = PARITY / "rr_parity"
for _p in (str(PARITY), str(BNG_PARITY), str(AMICI_PARITY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import _bng_common as bc  # noqa: E402
from rr_parity import _rr_common as rc  # noqa: E402

HERE = Path(__file__).resolve().parent
NETS_DIR = HERE / "nets"  # derived .net files (SBML -> .net via sbml_to_net)
OUT = HERE / "report_ode_engines_s4.json"

# rr_parity's curated SBML corpus + the figure-tied horizon manifest.
MODELS_ROOT = RR_PARITY / "models"
ODE_JOBS = RR_PARITY / "ode_jobs.json"

# The six largest curated SBML BioModels carrying a figure_sedml horizon, in table
# order 1..6 by ascending N (parallel to Table S3's small->large columns). ``label``
# is the display name (author-year, from each SBML ``<model name>``); ``row`` is the
# S4 column number. Files live at models/<model_id>/{xml,sedml}.
MODELS = [
    {"model_id": "BIOMD0000000834", "label": "Verma2016", "row": 1},
    {"model_id": "BIOMD0000000468", "label": "Koo2013", "row": 2},
    {"model_id": "BIOMD0000000559", "label": "Ouzounoglou2014", "row": 3},
    {"model_id": "BIOMD0000000667", "label": "Hornberg2005", "row": 4},
    {"model_id": "BIOMD0000000474", "label": "Smith2013", "row": 5},
    {"model_id": "BIOMD0000001065", "label": "vonDassow2000", "row": 6},
]

# Engine columns in report order.
ENGINE_ORDER = ["run_network", "bngsim", "roadrunner", "amici_klu", "amici_dense", "copasi"]

# run_network-vs-bngsim final-state agreement required to trust the run_network cell
# (the derived-.net path is novel to this table, so it is verified, not assumed).
PARITY_TOL = 1e-3


# --------------------------------------------------------------------------- #
# Normalized per-engine result (identical shape to S3).
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


def _final_state_mre(a, b):
    """Max relative final-state difference between two final-state vectors, aligned
    on their common leading species (both trace the SBML species order; sbml_to_net
    may append one trailing bookkeeping species, so a length mismatch aligns on the
    shorter prefix). A middle-inserted species could only inflate the mre, so this
    can false-*fail* but never false-pass. Returns the mre, or None if unusable."""
    if not a or not b:
        return None
    n = min(len(a), len(b))
    a = np.asarray(a[:n], dtype=float)
    b = np.asarray(b[:n], dtype=float)
    scale = np.maximum(np.abs(a), np.abs(b))
    mask = scale > 1e-9
    return float(np.max(np.abs(a - b)[mask] / scale[mask])) if mask.any() else 0.0


# --------------------------------------------------------------------------- #
# Engine adapters. Each runs in a forked child (see run_in_child) and returns a
# JSON-serializable dict. RoadRunner / AMICI / COPASI reuse the canonical parity
# adapters (they already take an SBML path + horizon). BNGsim is local here (it
# ingests the SBML natively via from_sbml — the inverse of S3's from_net path);
# run_network reuses bc.run_network_ode on the derived .net.
# --------------------------------------------------------------------------- #
def eng_bngsim(sbml_path, hz):
    """One BNGsim ODE run over the native SBML via ``Model.from_sbml``. Mirrors
    ``bc.bn_ode_net`` exactly, only swapping the model constructor: ``from_sbml``
    load is the build; the lazy Jacobian derivation + RHS codegen happen INSIDE the
    first ``sim.run`` (already inside ``integrate_cold_sec``)."""
    import bngsim

    t0 = time.perf_counter()
    model = bngsim.Model.from_sbml(str(sbml_path))
    load_sec = time.perf_counter() - t0

    sim = bngsim.Simulator(model, method="ode")

    t1 = time.perf_counter()
    r = sim.run(
        t_span=(hz["t_start"], hz["t_end"]),
        n_points=hz["n_points"],
        rtol=hz["rtol"],
        atol=hz["atol"],
    )
    cold_sec = time.perf_counter() - t1
    v = np.asarray(r.species)
    if not np.isfinite(v).all():
        return _err("bngsim produced a non-finite trajectory")
    warm: list[float] = []
    for _ in range(rc._warm_rep_count(cold_sec)):
        try:
            model.reset()
            t1 = time.perf_counter()
            sim.run(
                t_span=(hz["t_start"], hz["t_end"]),
                n_points=hz["n_points"],
                rtol=hz["rtol"],
                atol=hz["atol"],
            )
            warm.append(time.perf_counter() - t1)
        except Exception:
            break
    integ = rc._integrate_stats(cold_sec, warm)

    stats = r.solver_stats if hasattr(r, "solver_stats") else {}
    ls_code = (stats or {}).get("linear_solver", 0)
    config = {
        "codegen": sim.codegen_backend,
        "jacobian": sim.jacobian_strategy,
        "linear_solver": bc.LINEAR_SOLVER_NAMES.get(ls_code, f"kind_{ls_code}"),
        "cached": sim.codegen_cache_hit,
    }
    bd = {
        "load_sec": round(load_sec, 6),
        "jac_derive_sec": round(float(sim.last_jacobian_sec), 6),
        "codegen_sec": round(float(sim.last_codegen_sec), 6),
    }
    # final-time species vector (SBML/network order) for the run_network parity guard
    fs = [float(x) for x in np.asarray(v)[-1]]
    return _norm(
        float(load_sec),
        bd,
        config.get("cached"),
        integ,
        config,
        int(v.shape[1]),
        extra={"final_state": fs},
    )


def bn_net_final(net_path, hz):
    """bngsim solving the *derived* ``.net`` — the reference for "what the .net
    encodes", used to test conversion faithfulness (from_net vs from_sbml) and to
    attribute a run_network mismatch to the conversion vs the engine. Returns only
    the final-state vector."""
    t, v, names, tm = bc.bn_ode_net(
        net_path, hz["t_start"], hz["t_end"], hz["n_points"], hz["rtol"], hz["atol"]
    )
    v = np.asarray(v)
    if not np.isfinite(v).all():
        return _err("bngsim from_net produced a non-finite trajectory")
    return {"status": "ok", "final_state": [float(x) for x in v[-1]], "n_species": int(v.shape[1])}


def eng_run_network(net_path, hz, rn_bin):
    work = Path(tempfile.mkdtemp(prefix="s4rn_"))
    try:
        t, v, names, tm = bc.run_network_ode(
            net_path,
            rn_bin,
            t_start=hz["t_start"],
            t_end=hz["t_end"],
            n_steps=hz["n_steps"],
            rtol=hz["rtol"],
            atol=hz["atol"],
            out_prefix=str(work / "rn"),
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
    fs = [float(x) for x in np.asarray(v)[-1]]  # for the bngsim parity guard
    return _norm(
        0.0,
        bd,
        None,
        tm,
        tm.get("config", {}),
        n,
        extra={"n_calls": tm.get("n_calls"), "final_state": fs},
    )


def eng_roadrunner(sbml_path, hz):
    t, v, names, tm = rc.rr_ode(
        str(sbml_path), hz["t_start"], hz["t_end"], hz["n_points"], hz["rtol"], hz["atol"]
    )
    if not np.isfinite(v).all():
        return _err("roadrunner produced a non-finite trajectory")
    n = int(v.shape[1])
    build = float(tm.get("parse_interpret_codegen_sec") or 0.0)
    bd = {
        k: tm.get(k)
        for k in (
            "read_sec",
            "parse_sec",
            "interpret_sec",
            "codegen_sec",
            "jit_sec",
            "parse_interpret_codegen_sec",
            "model_cache_hit",
        )
    }
    cfg = tm.get("config", {})
    return _norm(build, bd, cfg.get("cached"), tm, cfg, n)


def eng_amici(sbml_path, hz, fresh):
    """Time BOTH AMICI linear-solver variants off a SINGLE compile. Returns a dict
    with keys ``amici_klu`` and ``amici_dense`` (plus ``_note``). The C++ compile is
    the shared one-time build recorded in both rows' ``build_sec`` (paid once)."""
    import _amici_common as amc
    import amici  # noqa: F401
    import amici.sim.sundials as ss

    sbml_str = Path(sbml_path).read_text()
    if fresh:
        key = hashlib.sha256((sbml_str + amc._BUILD_FLAGS).encode()).hexdigest()[:16]
        shutil.rmtree(amc.AMICI_CACHE / f"amici_{key}", ignore_errors=True)

    model, bt, cached = amc._build_model(sbml_str)
    build = (
        bt["parse_sec"]
        + bt["interpret_sec"]
        + bt["jac_derive_sec"]
        + bt["codegen_sec"]
        + bt["compile_sec"]
        + bt["load_sec"]
    )
    ts = np.linspace(hz["t_start"], hz["t_end"], hz["n_points"])
    sens_none = getattr(ss, "SensitivityOrder_none", 0)

    out = {
        "_note": (
            "AMICI codegen+compile is one-time and shared by amici_klu and "
            "amici_dense (paid once); build_sec repeats it in both rows."
        )
    }
    for eng, enumval in (
        ("amici_klu", ss.LinearSolver_KLU),
        ("amici_dense", ss.LinearSolver_dense),
    ):
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
                out[eng] = {
                    **_err(f"AMICI status {rd.status} (cold)", "error"),
                    "build_sec": round(build, 6),
                    "build_breakdown": bt,
                }
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
            cfg = {
                "codegen": "C++ (compiled)",
                "jacobian": "analytical (symbolic)",
                "linear_solver": ls,
                "cached": cached,
            }
            out[eng] = _norm(
                build,
                bt,
                cached,
                integ,
                cfg,
                n,
                extra={"integrate_cpu_ms": round(float(rd.cpu_time), 4)},
            )
        except Exception as e:  # noqa: BLE001
            out[eng] = {
                **_err(f"{type(e).__name__}: {e}"),
                "build_sec": round(build, 6),
                "build_breakdown": bt,
            }
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
        cfg = {
            "codegen": "interpreted (COPASI)",
            "jacobian": "internal",
            "linear_solver": "COPASI deterministic (LSODA)",
            "cached": None,
        }
        return _norm(build, {"import_config_sec": round(build, 6)}, None, integ, cfg, n)
    finally:
        with contextlib.suppress(Exception):
            COPASI.CRootContainer.removeDatamodel(dm)


# --------------------------------------------------------------------------- #
# Fork isolation + wall timeout (verbatim from S3): each engine call runs in a
# child so a native crash or hang is contained and reported, never fatal. Timing
# is measured INSIDE the child, so fork/pipe overhead is not charged.
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


import re  # noqa: E402


def _normalize_net_functions(net_path: Path) -> int:
    """Strip internal whitespace from every ``.net`` function EXPRESSION so the
    legacy ``run_network`` binary can parse it.

    ``sbml_to_net`` emits its synthetic reactant-guard fluxes as
    ``if((X) > 1e-300, flux / (X), 0)`` (spaces around operators and after commas).
    BNG2.pl's own expression handling tolerates that, but the standalone
    ``run_network`` CVODE binary's ``if()`` implementation is whitespace-intolerant
    (any space inside the ``if(...)`` argument list raises muParser
    "Missing parenthesis") — so the network is faithful yet unreadable by the
    legacy engine as written. muParser ignores whitespace in ordinary expressions
    and every token in a ``.net`` function body is a space-free identifier, number,
    or operator, so compressing the expression whitespace is semantics-preserving
    (verified: the guarded flux integrates identically). Returns the number of
    function lines rewritten. Idempotent."""
    text = net_path.read_text()
    lines = text.splitlines(keepends=True)
    in_fns = False
    changed = 0
    # "<idx> <name>(<sig>) <expression>" — compress spaces in <expression> only.
    fn_re = re.compile(r"^(\s*\d+\s+[A-Za-z_]\w*\([^)]*\)\s+)(.*?)(\s*)$")
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("begin functions"):
            in_fns = True
            continue
        if s.startswith("end functions"):
            in_fns = False
            continue
        if not in_fns or not s:
            continue
        m = fn_re.match(ln.rstrip("\n"))
        if not m:
            continue
        prefix, expr, _ = m.groups()
        compressed = re.sub(r"\s+", "", expr)
        if compressed != expr:
            nl = "\n" if ln.endswith("\n") else ""
            lines[i] = prefix + compressed + nl
            changed += 1
    if changed:
        net_path.write_text("".join(lines))
    return changed


def _disambiguate_net_names(net_path: Path) -> list[str]:
    """Rename ``.net`` PARAMETER definitions that collide with a FUNCTION of the
    same name, so the legacy ``run_network`` resolves bare references the way
    bngsim (and RoadRunner) do — to the function.

    ``sbml_to_net`` names its synthetic per-reaction flux helpers after the SBML
    reaction ids (``v1``, ``v13``, …) and *also* emits a same-named parameter
    (the reaction's initial flux, ``0.0``). A bare ``v1`` in a downstream function
    body is then ambiguous: bngsim's ``.net`` reader binds it to the function (the
    intended flux), but ``run_network`` binds it to the parameter (``0.0``),
    silently zeroing that reaction and producing a wrong-but-finite trajectory.
    The colliding parameters are the reaction-flux shadows and are not referenced
    as parameters anywhere (verified: renaming them leaves ``run_network`` matching
    bngsim to <1e-5), so renaming the parameter *definitions* only (no reference
    fixups) disambiguates in favor of the function — the reference interpretation.
    Returns the list of renamed names. Idempotent. (Upstream fix belongs in
    ``sbml_to_net``: namespace the synthetic flux functions / drop the shadow
    parameter — reported separately.)"""
    text = net_path.read_text()
    lines = text.splitlines(keepends=True)

    fn_names: set[str] = set()
    in_fns = False
    for ln in lines:
        s = ln.strip()
        if s.startswith("begin functions"):
            in_fns = True
            continue
        if s.startswith("end functions"):
            in_fns = False
            continue
        if in_fns and s:
            m = re.match(r"\s*\d+\s+([A-Za-z_]\w*)\(", ln)
            if m:
                fn_names.add(m.group(1))

    renamed: list[str] = []
    in_par = False
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("begin parameters"):
            in_par = True
            continue
        if s.startswith("end parameters"):
            in_par = False
            continue
        if not in_par or not s:
            continue
        m = re.match(r"(\s*\d+\s+)([A-Za-z_]\w*)(\s+.*)", ln.rstrip("\n"))
        if m and m.group(2) in fn_names:
            nl = "\n" if ln.endswith("\n") else ""
            lines[i] = f"{m.group(1)}__s4shadow_{m.group(2)}{m.group(3)}{nl}"
            renamed.append(m.group(2))
    if renamed:
        net_path.write_text("".join(lines))
    return renamed


# --------------------------------------------------------------------------- #
# SBML -> .net conversion via bngsim's own converter (the inverse of S3's OMEX
# export). Cached per model. Runs in the caller's process (bngsim); wrap the call
# in run_in_child at the call site for a timeout.
# --------------------------------------------------------------------------- #
def ensure_net(sbml_path, sedml_path, san, gate, hz):
    """Convert the curated SBML to a ``.net`` (``bngsim.sbml_to_net``) so
    run_network can integrate it. ``validate="L2"``, ``strict=True`` — an SBML the
    flat ``.net`` cannot carry faithfully (events, unsupported forcing) raises
    ``ConversionError`` and is reported ``status="unconvertible"`` (an empty
    run_network cell, e.g. BIOMD0000000474's four events). Passes ``sedml=`` so the
    figure horizon rides along. Returns the ``.net`` path + conversion provenance."""
    from bngsim import sbml_to_net
    from bngsim._exceptions import ConversionError

    NETS_DIR.mkdir(parents=True, exist_ok=True)
    net_out = NETS_DIR / f"{san}.net"
    if net_out.exists():
        return {
            "status": "ok",
            "net_path": str(net_out),
            "convert_sec": None,
            "cached": True,
        }
    t0 = time.perf_counter()
    try:
        rep = sbml_to_net(
            str(sbml_path),
            str(net_out),
            validate=gate,
            strict=True,
            sedml=str(sedml_path) if sedml_path else None,
            t_span=(hz["t_start"], hz["t_end"]),
            n_points=hz["n_points"],
        )
    except ConversionError as e:
        with contextlib.suppress(Exception):
            net_out.unlink(missing_ok=True)
        return _err(f"sbml_to_net(strict) refused: {e}", "unconvertible")
    convert_sec = time.perf_counter() - t0
    # Adapt the faithful .net to the legacy run_network binary (formatting/naming
    # only; math unchanged, matching bngsim's reference interpretation of the same
    # .net — enforced downstream by the run_network-vs-bngsim parity guard):
    #  (1) strip whitespace the whitespace-intolerant if() parser rejects;
    #  (2) disambiguate synthetic flux parameter/function name collisions.
    n_norm = _normalize_net_functions(net_out)
    renamed = _disambiguate_net_names(net_out)
    return {
        "status": "ok",
        "net_path": str(net_out),
        "convert_sec": round(convert_sec, 4),
        "cached": False,
        "rhs_faithful": rep.rhs_faithful,
        "max_rhs_delta": rep.max_rhs_delta,
        "gate_ok": bool(rep.ok),
        "n_species": rep.n_species,
        "n_reactions": rep.n_reactions,
        "functions_normalized": n_norm,
        "params_disambiguated": renamed,
    }


# --------------------------------------------------------------------------- #
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


def sedml_initial_time(sedml_path, output_start, t_end):
    """SED-ML ``initialTime`` — the start of INTEGRATION, distinct from
    ``outputStartTime`` (the start of the plotted window). ode_jobs.json flattens the
    figure horizon to ``t_start`` = ``outputStartTime``, dropping ``initialTime``; but
    integration must run from ``initialTime`` so events before the output window fire
    (e.g. BIOMD834/Verma2016's ``H:=1.8`` step at t=200, window [400,700]). Reads the
    ``<uniformTimeCourse>`` whose output window matches this job; falls back to the
    first, then to 0.0. Returns the integration start time."""
    if not sedml_path or not Path(sedml_path).exists():
        return 0.0

    def _attr(tag, name):
        mm = re.search(rf'{name}\s*=\s*"([^"]*)"', tag)
        try:
            return float(mm.group(1)) if mm else None
        except ValueError:
            return None

    text = Path(sedml_path).read_text(errors="replace")
    first = None
    for m in re.finditer(r"<uniformTimeCourse\b[^>]*>", text):
        tag = m.group(0)
        it = _attr(tag, "initialTime")
        if it is None:
            continue
        os_, oe = _attr(tag, "outputStartTime"), _attr(tag, "outputEndTime")
        if (
            os_ is not None
            and oe is not None
            and abs(os_ - output_start) < 1e-6
            and abs(oe - t_end) < 1e-6
        ):
            return it
        if first is None:
            first = it
    return first if first is not None else 0.0


def resolve_models(args):
    """Attach each model's SBML + SED-ML file paths from the rr_parity corpus and
    its figure_sedml horizon from ode_jobs.json. The integration start is the SED-ML
    ``initialTime`` (not ``outputStartTime``); they differ only for Verma2016 here."""
    jobs = {j["model_id"]: j for j in json.loads(ODE_JOBS.read_text())["jobs"]}
    picked = []
    for spec in MODELS:
        mid = spec["model_id"]
        job = jobs.get(mid)
        if job is None:
            print(f"  WARN: {mid} not in ode_jobs.json — skipped")
            continue
        p = job["params"]
        xml = MODELS_ROOT / mid / Path(job["model"]).name
        sedml_name = p.get("sedml")
        sedml = (MODELS_ROOT / mid / sedml_name) if sedml_name else None
        if not xml.exists():
            print(f"  WARN: SBML missing for {mid}: {xml} — skipped")
            continue
        output_start = float(p["t_start"])
        t_end = float(p["t_end"])
        init_t = sedml_initial_time(sedml, output_start, t_end)
        picked.append(
            {
                **spec,
                "sbml_path": xml,
                "sedml_path": sedml,
                "horizon_source": p.get("horizon_source"),
                "t_start": init_t,  # integration start = SED-ML initialTime
                "output_start": output_start,  # SED-ML outputStartTime (plotted window)
                "t_end": t_end,
                "n_points": int(p["n_points"]),
                "rtol": float(p["rtol"]),
                "atol": float(p["atol"]),
            }
        )
    if args.models:
        subs = [s.strip() for s in args.models.split(",") if s.strip()]
        picked = [p for p in picked if any(s in p["model_id"] or s in p["label"] for s in subs)]
    if args.limit:
        picked = picked[: args.limit]
    return picked


def horizon_for(spec, args):
    """Per-model figure_sedml horizon (rtol/atol/tol overridable via CLI). Integration
    runs [t_start, t_end] = [SED-ML initialTime, outputEndTime] so pre-window events
    fire; ``output_start`` records the SED-ML outputStartTime (the plotted window).
    n_steps is n_points-1 so run_network's (step_size, n_steps) grid matches the
    sampled points the SBML engines use."""
    n_points = int(spec["n_points"])
    return {
        "t_start": float(spec["t_start"]),
        "output_start": float(spec.get("output_start", spec["t_start"])),
        "t_end": float(spec["t_end"]),
        "n_points": n_points,
        "n_steps": max(1, n_points - 1),
        "rtol": args.rtol if args.rtol is not None else float(spec["rtol"]),
        "atol": args.atol if args.atol is not None else float(spec["atol"]),
        "rn_timeout": args.timeout,
        "horizon_source": spec["horizon_source"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--models", default="", help="Comma-separated model_id/label substring filter."
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--redo", action="store_true", help="Re-time models already present in the report."
    )
    ap.add_argument(
        "--fresh-amici",
        action="store_true",
        help="Clear each model's AMICI compile cache first so build_sec is a true cold compile.",
    )
    ap.add_argument(
        "--convert-gate",
        default="L2",
        choices=["L1", "L2", "full", "none"],
        help="sbml_to_net validation gate for the run_network .net (default L2).",
    )
    ap.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="Override each model's figure_sedml rtol (default: per-model from SED-ML).",
    )
    ap.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Override each model's figure_sedml atol (default: per-model from SED-ML).",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=1200.0,
        help="Per-engine wall timeout (s); also the run_network subprocess timeout.",
    )
    ap.add_argument("--convert-timeout", type=float, default=1200.0)
    ap.add_argument(
        "--engines",
        default=",".join(ENGINE_ORDER),
        help="Comma-separated subset of engines to run.",
    )
    args = ap.parse_args()

    rn_bin = resolve_run_network()
    want_engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    models = resolve_models(args)

    prior = {}
    if OUT.exists():
        with contextlib.suppress(Exception):
            prior = {r["model_id"]: r for r in json.loads(OUT.read_text()).get("results", [])}

    print("=" * 74)
    print("  Table S4 — cross-engine ODE timing (6 largest SBML BioModels x 6 engines)")
    print("=" * 74)
    print(f"  models: {len(models)}   engines: {want_engines}")
    print(f"  run_network: {rn_bin}")
    print(
        f"  fresh-amici: {args.fresh_amici}   convert-gate: {args.convert_gate}   "
        f"timeout: {args.timeout:g}s"
    )
    print(f"  bngsim: {_versions().get('bngsim')}")
    print()

    results = dict(prior)
    for spec in models:
        mid = spec["model_id"]
        if not args.redo and mid in prior and prior[mid].get("_complete"):
            print(f"  [skip] {spec['label']:20} (already complete)")
            continue
        san = mid.replace("/", "__")
        sbml_path = spec["sbml_path"]
        sedml_path = spec["sedml_path"]
        hz = horizon_for(spec, args)
        owin = (
            f" output[{hz['output_start']:g},{hz['t_end']:g}]"
            if hz["output_start"] != hz["t_start"]
            else ""
        )
        print(
            f"  #{spec['row']} {spec['label']:18} {mid}  integrate "
            f"[{hz['t_start']:g},{hz['t_end']:g}]{owin} n_points={hz['n_points']} "
            f"tol={hz['rtol']:g}/{hz['atol']:g} ({hz['horizon_source']})"
        )

        # SBML -> .net conversion — needed by run_network only. Fork-isolated.
        net_path = None
        net_conv = None
        if "run_network" in want_engines:
            net_conv = run_in_child(
                lambda sbml_path=sbml_path, sedml_path=sedml_path, san=san, hz=hz: ensure_net(
                    sbml_path, sedml_path, san, args.convert_gate, hz
                ),
                args.convert_timeout,
            )
            if net_conv.get("status") == "ok":
                net_path = net_conv["net_path"]
                tag = "cached" if net_conv.get("cached") else f"{net_conv.get('convert_sec')}s"
                print(
                    f"       sbml->net: {tag}  rhs_faithful={net_conv.get('rhs_faithful')} "
                    f"gate_ok={net_conv.get('gate_ok')}"
                )
            else:
                print(
                    f"       sbml->net: {net_conv.get('status').upper()} ({net_conv.get('error')})"
                )

        engines: dict[str, dict] = {}
        for eng in want_engines:
            if eng == "run_network":
                if net_path is None:
                    res = _err(
                        f"no .net (sbml_to_net {(net_conv or {}).get('status')}): "
                        f"{(net_conv or {}).get('error')}",
                        (net_conv or {}).get("status", "skipped"),
                    )
                else:
                    res = run_in_child(
                        lambda net_path=net_path, hz=hz: eng_run_network(net_path, hz, rn_bin),
                        args.timeout,
                    )
            elif eng == "bngsim":
                res = run_in_child(
                    lambda sbml_path=sbml_path, hz=hz: eng_bngsim(sbml_path, hz), args.timeout
                )
            elif eng == "roadrunner":
                res = run_in_child(
                    lambda sbml_path=sbml_path, hz=hz: eng_roadrunner(sbml_path, hz), args.timeout
                )
            elif eng == "copasi":
                res = run_in_child(
                    lambda sbml_path=sbml_path, hz=hz: eng_copasi(sbml_path, hz), args.timeout
                )
            elif eng in ("amici_klu", "amici_dense"):
                if "amici_klu" in engines or "amici_dense" in engines:
                    res = engines.get(eng)  # already produced below
                else:
                    amici_out = run_in_child(
                        lambda sbml_path=sbml_path, hz=hz: eng_amici(
                            sbml_path, hz, args.fresh_amici
                        ),
                        args.timeout,
                    )
                    note = amici_out.get("_note") if isinstance(amici_out, dict) else None
                    for k in ("amici_klu", "amici_dense"):
                        sub = amici_out.get(k) if isinstance(amici_out, dict) else None
                        engines[k] = (
                            sub
                            if sub is not None
                            else dict(
                                amici_out
                                if isinstance(amici_out, dict)
                                else _err("amici child failed")
                            )
                        )
                        if note and isinstance(engines[k], dict):
                            engines[k].setdefault("_note", note)
                    res = engines.get(eng)
            else:
                res = _err(f"unknown engine {eng}", "skipped")
            if res is not None:
                engines[eng] = res
            _print_engine(eng, engines.get(eng))

        # ---- run_network validity: a two-stage check ----------------------------
        # run_network integrates a .net *derived* from the SBML (sbml_to_net) — a
        # path never validated before this table — so its cell is trusted only if:
        #   (1) the derived .net is FAITHFUL to the SBML: bngsim solving the .net
        #       (bn_net_final) matches bngsim solving the SBML (from_sbml). Some SBML
        #       constructs (cross-compartment volume scaling, assignment/rate-rule
        #       forcing) survive sbml_to_net's own probe check yet diverge on
        #       integration — an honest "not faithfully convertible" that must NOT be
        #       blamed on run_network's speed; and
        #   (2) run_network then reproduces that faithful trajectory (from_sbml).
        # Both compare final-state vectors in SBML species order. A mismatch marks
        # the cell (unfaithful_net vs parity_fail) so a wrong-but-finite run_network
        # number is never tabulated.
        rn = engines.get("run_network")
        bn_sbml = engines.get("bngsim")
        if (
            net_conv
            and net_conv.get("status") == "ok"
            and isinstance(bn_sbml, dict)
            and bn_sbml.get("final_state")
            and net_path
        ):
            bn_net = run_in_child(
                lambda net_path=net_path, hz=hz: bn_net_final(net_path, hz), args.timeout
            )
            faithful = None
            if isinstance(bn_net, dict) and bn_net.get("final_state"):
                fm = _final_state_mre(bn_net["final_state"], bn_sbml["final_state"])
                if fm is not None:
                    net_conv["net_faithful_mre"] = fm
                    net_conv["net_faithful"] = faithful = fm <= PARITY_TOL
                    print(f"       net faithful:  {faithful} (from_net vs from_sbml mre={fm:.2e})")
            if isinstance(rn, dict):
                rn["net_faithful"] = faithful
                if faithful is False:
                    # The .net does not reproduce the SBML; no run_network timing on it
                    # is meaningful. Root cause is the conversion, not the engine.
                    rn["status"] = "unfaithful_net"
                    rn["error"] = (
                        f"derived .net does not reproduce the SBML "
                        f"(bngsim from_net vs from_sbml mre={net_conv.get('net_faithful_mre'):.2e}); "
                        "run_network timing excluded"
                    )
                    print("       run_network    UNFAITHFUL_NET (conversion, not engine)")
                elif rn.get("status") == "ok" and rn.get("final_state"):
                    pm = _final_state_mre(rn["final_state"], bn_sbml["final_state"])
                    rn["parity_vs_bngsim_mre"] = pm
                    rn["parity_ok"] = None if pm is None else pm <= PARITY_TOL
                    if pm is not None and pm > PARITY_TOL:
                        rn["status"] = "parity_fail"
                        rn["error"] = (
                            f"run_network disagrees with bngsim on a faithful .net "
                            f"(mre={pm:.2e} > {PARITY_TOL:g})"
                        )
                        print(f"       run_network    PARITY_FAIL mre={pm:.2e}")
                    elif pm is not None:
                        print(f"       run_network    parity_ok mre={pm:.2e}")
        # final_state fingerprints served the guards; drop them from the persisted JSON.
        for e in engines.values():
            if isinstance(e, dict):
                e.pop("final_state", None)

        # Canonical N/reactions: BNGsim's from_sbml network (fall back to the .net
        # conversion counts, then to any engine that reported n_species).
        n_species = (engines.get("bngsim") or {}).get("n_species")
        if n_species is None and net_conv and net_conv.get("status") == "ok":
            n_species = net_conv.get("n_species")
        n_reactions = (net_conv or {}).get("n_reactions") if net_conv else None
        if n_species is None:
            for e in engines.values():
                if isinstance(e, dict) and e.get("n_species"):
                    n_species = e["n_species"]
                    break

        results[mid] = {
            "model_id": mid,
            "label": spec["label"],
            "row": spec["row"],
            "n_species": n_species,
            "n_reactions": n_reactions,
            "horizon": hz,
            "net_conversion": net_conv,
            "engines": engines,
            "_complete": all(e in engines for e in want_engines),
        }
        _write(results, args, want_engines)
        print()

    print(f"  report: {OUT}")
    return 0


def _fmt(x, ms=False):
    if not isinstance(x, (int, float)):
        return "   --   "
    return f"{x * 1000:8.2f}ms" if ms else f"{x:8.3f}s"


def _print_engine(eng, r):
    if not r:
        print(f"       {eng:14} (none)")
        return
    if r.get("status") != "ok":
        print(f"       {eng:14} {r.get('status').upper()}: {(r.get('error') or '')[:60]}")
        return
    ls = (r.get("config") or {}).get("linear_solver", "")
    print(
        f"       {eng:14} cold={_fmt(r.get('cold_total_sec'))} "
        f"(build={_fmt(r.get('build_sec'))}+first={_fmt(r.get('integrate_cold_sec'))}) "
        f"warm={_fmt(r.get('warm_sec'), ms=True)}  {ls}"
    )


def _write(results, args, want_engines):
    ordered = sorted(results.values(), key=lambda r: (r.get("row") or 0))
    doc = {
        "_meta": {
            "suite": "ode_engines_s4_sbml",
            "description": (
                "Cross-engine ODE timing on the six largest curated SBML BioModels "
                "carrying a real figure-tied horizon (the SBML counterpart to Table "
                "S3). Cold = one-time build + first solve; warm = median of "
                "subsequent solves. SBML engines (RoadRunner/AMICI/COPASI) and "
                "BNGsim ingest the curated SBML natively; run_network integrates a "
                ".net derived offline via bngsim.sbml_to_net (model-prep, not "
                "charged; recorded in net_conversion)."
            ),
            "engines": want_engines,
            "metric": {
                "cold": "build_sec + integrate_cold_sec (= cold_total_sec)",
                "warm": "integrate_warm_median_sec (= warm_sec)",
                "warm_reps": {"max": rc.WARM_REPS, "budget_sec": rc.WARM_BUDGET_SEC},
            },
            "horizon_source": "figure_sedml (rr_parity ode_jobs.json, per-model)",
            "horizon_note": (
                "integration span = [SED-ML initialTime, outputEndTime] so pre-window "
                "events fire; output_start = SED-ML outputStartTime (plotted window)."
            ),
            "tol_override": {"rtol": args.rtol, "atol": args.atol},
            "convert_gate": args.convert_gate,
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
