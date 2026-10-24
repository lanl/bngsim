"""Shared machinery for the rr_parity suite (engine adapters + isolation).

Both ``rr_run.py`` (the cross-engine report) and ``rr_golden.py`` (the
bngsim-only golden emitter) drive the SAME bngsim + RoadRunner calls and the
SAME process-isolation scheduler, so they live here once. Everything is
self-contained to the suite — nothing is imported from ``harness/`` or
``benchmarks/`` (those are ported here verbatim where needed).

Two facts that shape the adapters, both verified directly and documented in
``dev/notes/SBML_VS_ROADRUNNER.md``:

  * **Both engines report CONCENTRATION.** bngsim's ``Result.species`` is
    amount/volume and RoadRunner's default output columns are ``[S]``
    concentrations, so concentration-vs-concentration is dimensionally
    consistent for any compartment volume. (Selecting RR *amounts* would
    mismatch bngsim on every V≠1 compartment.)
  * **RoadRunner reports only FLOATING species** while bngsim reports every
    species (boundary/constant included). So comparison is always over the
    *intersection* of species names (``align_common``), matched by name, not
    by column order.

Isolation matters: RoadRunner can segfault on pathological SBML, so every job
runs in its own spawned subprocess that the parent kills on wall-clock
overrun. One bad model never takes down the screen.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import queue as _queue
import time
from pathlib import Path

import numpy as np

# Shared tight integration tolerance forced on BOTH engines for ODE, by both the
# parity run (rr_run) and golden generation (rr_golden) so the golden trajectory
# is computed at exactly the tolerance the parity check validates. The SBML
# analog of bng_parity's shared atol/rtol=1e-8 default, but tighter: BioModels
# run in concentration space (species ~1e-5..1e-9) where a 1e-8 atol swamps the
# signal. The per-job SED-ML rtol/atol stay in the manifest as provenance only.
DEFAULT_RTOL = 1e-9
DEFAULT_ATOL = 1e-12

# Integration timing: the first solve is COLD (one-time CVODE workspace +
# linear-solver alloc + RHS .so/ExprTk page-fault + CPU ramp); subsequent solves
# reuse the warm solver. We time a cold solve, then up to WARM_REPS warm solves and
# report the spread. The warm MIN is the marginal per-integration cost fitting/MCMC
# would pay if a Simulator were reused; COLD is the full first-solve cost a caller
# pays when it builds a fresh Simulator per evaluation. The warm loop is budget-capped so
# a stiff model never multiplies the wall budget: n_warm = min(WARM_REPS, ⌊budget/cold⌋).
WARM_REPS = 5
WARM_BUDGET_SEC = 3.0


def _integrate_stats(cold_sec: float, warm: list) -> dict:
    """Cold + warm integration timing for one engine/model. ``warm`` is the list of
    measured warm-solve wall times (may be empty for a very slow model that the
    budget cap skipped). Returns flat seconds: ``integrate_sec`` (the headline =
    warm-min, or cold when no warm reps ran), plus cold/min/max/median/n for display.
    """
    import statistics as _st

    headline = min(warm) if warm else cold_sec
    return {
        "integrate_sec": round(headline, 6),
        "integrate_cold_sec": round(cold_sec, 6),
        "integrate_warm_min_sec": round(min(warm), 6) if warm else None,
        "integrate_warm_max_sec": round(max(warm), 6) if warm else None,
        "integrate_warm_median_sec": round(_st.median(warm), 6) if warm else None,
        "integrate_n_warm": len(warm),
    }


def _warm_rep_count(cold_sec: float) -> int:
    """Adaptive warm-rep count: full WARM_REPS for fast models, fewer (down to 0)
    as the cold solve approaches the budget, so a stiff model never runs 5×."""
    return max(0, min(WARM_REPS, int(WARM_BUDGET_SEC / max(cold_sec, 1e-6))))


# LinearSolverKind (include/bngsim/result.hpp) → display name for the report.
# Pinned by T0.4: 1 is KLU, NOT "Band" (the old harness guess).
LINEAR_SOLVER_NAMES = {0: "Dense", 1: "KLU", 2: "LAPACK-dense"}


def hardware_info() -> dict:
    """Best-effort CPU / OS identification so the report's timing numbers are
    interpretable and reproducible. Returns ``{cpu, physical_cores, logical_cores,
    platform}`` — facts only; the interpretation caveat (timings collected under
    N-way concurrency) is added by the renderer, not stored here."""
    import os
    import platform
    import subprocess

    cpu = platform.processor() or ""
    physical = None
    try:
        sysname = platform.system()
        if sysname == "Darwin":
            cpu = (
                subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
                or cpu
            )
            physical = int(
                subprocess.run(
                    ["sysctl", "-n", "hw.physicalcpu"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
            )
        elif sysname == "Linux":
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        cpu = line.split(":", 1)[1].strip()
                        break
    except Exception:
        pass
    return {
        "cpu": cpu or "unknown",
        "physical_cores": physical,
        "logical_cores": os.cpu_count(),
        "platform": platform.platform(),
    }


def sundials_version() -> str | None:
    """The SUNDIALS (CVODE) version both engines' integrator is built against,
    read once from the vendored ``sundials_config.h`` that ships in the bngsim
    wheel. ``SUNDIALS_VERSION`` is a compile-time macro with no runtime accessor,
    so we read the header rather than guess. Returns e.g. ``"7.2.1"``, or ``None``
    if the header can't be located. Recorded in the report ``_meta["versions"]``
    so the matrix can name the exact integrator build without hardcoding it.
    """
    import re
    import site

    import bngsim

    candidates = [
        Path(bngsim.__file__).resolve().parent / "include" / "sundials" / "sundials_config.h"
    ]
    for sp in (*site.getsitepackages(), site.getusersitepackages()):
        candidates.append(Path(sp) / "bngsim" / "include" / "sundials" / "sundials_config.h")
    for hdr in candidates:
        try:
            if hdr.is_file():
                m = re.search(r'#define\s+SUNDIALS_VERSION\s+"([^"]+)"', hdr.read_text())
                if m:
                    return m.group(1)
        except OSError:
            continue
    return None


# Cumulative effort tiers (low ⊂ medium ⊂ high). Ported from benchmarks/_effort
# so the suite carries no benchmarks dependency. SSA jobs carry a tier in
# params["effort"]; ODE jobs carry none and are never effort-filtered.
EFFORT_LEVELS = ("low", "medium", "high")


def effort_allows(selected: str, tier: str | None) -> bool:
    """True if a job tagged ``tier`` runs at the ``selected`` effort level.

    A job with no tier (every ODE job) always runs — effort only prunes the
    cost-tiered SSA jobs.
    """
    if tier is None:
        return True
    return EFFORT_LEVELS.index(tier) <= EFFORT_LEVELS.index(selected)


def set_rr_quiet() -> None:
    """Silence libRoadRunner's chatter (it logs warnings to stderr by default)."""
    try:
        import roadrunner

        roadrunner.Logger.setLevel(roadrunner.Logger.LOG_FATAL)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Column alignment
# --------------------------------------------------------------------------- #
def align_common(
    bn_names: list[str], rr_names: list[str]
) -> tuple[list[int], list[int], list[str]] | None:
    """Return (bn_idx, rr_idx, common_names) over the shared species, or None.

    Both engines read the same SBML, so the species *id* sets must overlap, but
    RoadRunner emits only floating species and neither engine's column order is
    assumed. None means a fully disjoint species set — a structural loader
    divergence the caller surfaces loudly (never a silent pass).
    """
    bn_map = {n: i for i, n in enumerate(bn_names)}
    rr_map = {n: i for i, n in enumerate(rr_names)}
    common = sorted(set(bn_map) & set(rr_map))
    if not common:
        return None
    return [bn_map[n] for n in common], [rr_map[n] for n in common], common


# --------------------------------------------------------------------------- #
# Per-process warmup (the third cost type, alongside per-model and per-integration)
# --------------------------------------------------------------------------- #
def measure_warmup(include_roadrunner: bool = True) -> dict:
    """One-time per-process warmup cost for both engines.

    Must be called **once at worker start, before any engine call** so the SymPy
    import is captured here as warmup rather than charged to the first model's load
    (it is lazily imported on bngsim's analytical-Jacobian path). Each rr_parity job
    runs in its own subprocess, so each job yields one warmup sample; the matrix
    aggregates them (mean/median/std) in its header.

    - **bngsim:** the SymPy import + one trivial ``sp.diff`` (warms the diff
      machinery too, so the per-model ``last_jacobian_sec`` reflects pure
      derivation). This is the cost GH #145's lazy deferral targets.
    - **RoadRunner:** LLVM/JIT-engine init. Preferred from the instrumented RR
      engine getter when present; until that lands, an ``import roadrunner`` timing
      proxy, labelled in ``roadrunner_source`` so the matrix never conflates them.
    """
    t0 = time.perf_counter()
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr  # noqa: F401

    _ = sp.diff(sp.Symbol("x") ** 2, sp.Symbol("x"))
    bn_sec = time.perf_counter() - t0

    if not include_roadrunner:
        # A replay from a reference archive (issue #805) never loads RoadRunner.
        return {"bngsim_sec": round(bn_sec, 6)}
    rr_sec, rr_source = _rr_warmup()
    return {
        "bngsim_sec": round(bn_sec, 6),
        "roadrunner_sec": round(rr_sec, 6),
        "roadrunner_source": rr_source,
    }


def _rr_warmup() -> tuple[float, str]:
    """RoadRunner's one-time per-process warmup, shared by the ODE and SSA
    measures. The instrumented RoadRunner build exposes its import/init cost as a
    float attribute (``__warmup_sec__``, set once at import) — prefer it over our
    wall-around-import proxy, and label which source was used so the matrix never
    conflates them."""
    t0 = time.perf_counter()
    import roadrunner

    rr_import_sec = time.perf_counter() - t0
    rr_engine = getattr(roadrunner, "__warmup_sec__", None)
    if rr_engine is not None:
        return float(rr_engine), "engine"
    return rr_import_sec, "import-proxy"


def measure_ssa_warmup(include_roadrunner: bool = True) -> dict:
    """One-time per-process warmup cost for the SSA screen (both engines).

    The SSA taxonomy differs from the ODE one (see :func:`measure_warmup`):
    bngsim's SSA path derives NO analytical Jacobian and imports NO SymPy, so its
    warmup is the ``_bngsim_core`` extension load — measured here as the first
    ``import bngsim`` in the worker. RoadRunner's gillespie integrator still JITs,
    so its warmup is the same LLVM/JIT engine init the ODE path measures.

    Must be called **once at worker start, before any engine call**, so the
    extension import is captured here as per-process warmup rather than charged to
    the first model's per-model load.

    ``include_roadrunner=False`` (a replay from a reference archive, issue #805)
    measures bngsim alone and never imports RoadRunner.
    """
    t0 = time.perf_counter()
    import bngsim  # noqa: F401  — the _bngsim_core .so load is the measured cost

    bn_sec = time.perf_counter() - t0
    if not include_roadrunner:
        return {"bngsim_sec": round(bn_sec, 6)}
    rr_sec, rr_source = _rr_warmup()
    return {
        "bngsim_sec": round(bn_sec, 6),
        "roadrunner_sec": round(rr_sec, 6),
        "roadrunner_source": rr_source,
    }


def _ensemble_stats(rep_secs: list[float]) -> dict:
    """Per-replicate ensemble simulation timing for one engine (SSA).

    Unlike the ODE cold→warm-reuse split (ONE model, the same solve repeated on a
    warm solver), an SSA ensemble runs N **independent** replicates — the model is
    loaded once, then cloned/reset and reseeded per replicate — so there is no
    warm-reuse of a single integration. We report the replicate-time distribution
    (mean/median/min/max + total ensemble wall), plus the rep-1 COLD vs the warm
    (reps 2..N) median: the first trajectory still pays the one-time page-fault /
    CPU ramp that the rest reuse. ``rep_secs`` is the per-replicate simulation
    wall-times in seed order. Empty list → ``{"n_rep": 0}`` (a model with no
    timed replicate, e.g. an engine that raised before the loop)."""
    import statistics as _st

    if not rep_secs:
        return {"n_rep": 0}
    warm = rep_secs[1:]
    return {
        "n_rep": len(rep_secs),
        "rep_mean_sec": round(_st.mean(rep_secs), 6),
        "rep_median_sec": round(_st.median(rep_secs), 6),
        "rep_min_sec": round(min(rep_secs), 6),
        "rep_max_sec": round(max(rep_secs), 6),
        "rep_cold_sec": round(rep_secs[0], 6),
        "rep_warm_median_sec": round(_st.median(warm), 6) if warm else None,
        "ensemble_sec": round(sum(rep_secs), 6),
    }


# --------------------------------------------------------------------------- #
# bngsim adapters
# --------------------------------------------------------------------------- #
def bn_ode(
    xml: str,
    t_start: float,
    t_end: float,
    n_points: int,
    rtol: float,
    atol: float,
    force_dense_linear_solver: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """One deterministic bngsim run. Returns (time, values[n_time,n_sp], names, timing).

    ``t_span=(t_start, t_end)`` integrates from ``t_start`` (the model's initial
    state applied there) and samples ``n_points`` uniformly over ``[t_start,
    t_end]`` — matching ``rr.simulate(t_start, t_end, n_points)``, which likewise
    begins integration at ``t_start`` (it does NOT pre-integrate ``[0, t_start]``),
    so the two grids and trajectories match for ``t_start>0`` too.

    Callers pass the SED-ML ``initialTime`` as ``t_start`` so pre-``outputStartTime``
    dynamics/events fire (GH #19); the ``outputStartTime`` grid was the old,
    event-dropping behavior (BIOMD834/Verma2016's ``H:=1.8 @ t>=200`` event with
    window ``[400,700]`` never fired). For the common ``initialTime == outputStartTime``
    model this is unchanged.

    The timing dict carries a fine per-phase breakdown {"io_sec", "parse_sec",
    "interpret_sec", "jac_derive_sec", "codegen_sec", "integrate_sec",
    "parse_interpret_sec" (back-compat sum), "config"}. Each setup phase is timed
    at its own boundary by the engine — libSBML parse
    (Simulator.last_libsbml_parse_sec), doc→_core interpretation
    (last_interpret_sec), analytical-Jacobian derivation (last_jacobian_sec,
    bngsim-only — RR uses FD), and codegen (last_codegen_sec) — so the breakdown is
    non-overlapping with NO load−codegen subtraction. config = {"codegen",
    "jacobian", "linear_solver", "cached"} from the engine's public accessors. Uses
    an XML string to isolate file I/O. The one-time SymPy import is per-process
    warmup, measured by measure_warmup(), not charged here. A single run().
    """
    from pathlib import Path

    import bngsim
    from _core.work import work_counters

    # 1. File I/O (if xml is a path; SBML text itself can be too long for a
    # file name, which Path.exists() raises on)
    t0 = time.perf_counter()
    if not str(xml).lstrip().startswith("<") and Path(xml).exists():
        xml_string = Path(xml).read_text()
        io_sec = time.perf_counter() - t0
    else:
        # Already a string
        xml_string = xml
        io_sec = 0.0

    # 2. Parse + Interpret: XML string -> bngsim Model (libSBML parse + Python
    #    interpretation). For >=256-species models this also runs the auto-codegen
    #    cc compile. Each phase is timed at its own boundary inside the loader and
    #    read back via the engine accessors below — no wall-around-load subtraction.
    import bngsim._sbml_loader as sbml_loader

    model = sbml_loader.load_sbml_string(xml_string)

    sim = bngsim.Simulator(
        model, method="ode", force_dense_linear_solver=force_dense_linear_solver
    )

    # 3. Integrate: a single run (the .so / ExprTk RHS is already prepared at load,
    #    so this is integration + one-time CVODE setup — no codegen here).
    # Integration timing (cold + warm). The first solve is COLD (one-time CVODE
    # workspace/linear-solver alloc + RHS page-fault + CPU ramp) and its trajectory
    # feeds the parity verdict; then up to _warm_rep_count(cold) warm solves —
    # model.reset() restores IC each time (bngsim's run() continues from state) and
    # the solver workspace is reused → genuinely warm. warm-min is the marginal
    # per-integration cost; cold is the one-shot first-solve (fresh-Simulator) cost.
    t1 = time.perf_counter()
    r = sim.run(t_span=(t_start, t_end), n_points=n_points, rtol=rtol, atol=atol)
    cold_sec = time.perf_counter() - t1
    warm: list[float] = []
    for _ in range(_warm_rep_count(cold_sec)):
        # Warm timing is best-effort and MUST NOT change the verdict: the cold run
        # above already produced the trajectory the parity check uses. A warm
        # re-solve that raises (e.g. CVODE step-limit on a re-integration) just
        # stops the timing loop — it is never re-raised as an engine failure.
        try:
            model.reset()
            t1 = time.perf_counter()
            sim.run(t_span=(t_start, t_end), n_points=n_points, rtol=rtol, atol=atol)
            warm.append(time.perf_counter() - t1)
        except Exception:
            break
    integ = _integrate_stats(cold_sec, warm)

    # Per-model setup phases, each timed at its own boundary by the engine
    # (T-instr): libSBML parse · interpret · analytical-Jacobian derivation ·
    # codegen. No load−codegen subtraction — every phase is explicit. load_sec
    # (the wall around load_sbml_string) is retained only as a cross-check below.
    libsbml_parse_sec = sim.last_libsbml_parse_sec
    interpret_sec = sim.last_interpret_sec
    jac_derive_sec = sim.last_jacobian_sec
    codegen_sec = sim.last_codegen_sec

    # Config: what the engine ACTUALLY chose, straight from its public accessors
    # (T0.1/T0.2/T0.4) — no guessing from private attributes:
    #   codegen — the real RHS backend ("exprtk"/"cc"/"mir"); MIR is no longer
    #             mislabeled ExprTk.
    #   jacobian — the resolved strategy ("analytical"/"fd"/"jax"), not the
    #             requested mode.
    #   linear_solver — the LinearSolverKind name (1 ⇒ "KLU", never "Band").
    #   cached — DEFINITIVE (not inferred from wall time): the engine reports at
    #             its get_cached_so branch whether the .so was reused from the
    #             on-disk cache (True), compiled fresh (False), or no .so was
    #             involved (None — ExprTk or MIR).
    stats = r.solver_stats if hasattr(r, "solver_stats") else {}
    ls_code = stats.get("linear_solver", 0)
    config = {
        "codegen": sim.codegen_backend,
        "jacobian": sim.jacobian_strategy,
        "linear_solver": LINEAR_SOLVER_NAMES.get(ls_code, f"kind_{ls_code}"),
        # GH #132 adaptive gate: of the dense factorizations, how many took the
        # BLAS dgetrf path vs the built-in dense LU. Only nonzero on LAPACK-dense
        # (kind 2) runs that crossed the K gate; a LAPACK-dense run with
        # dense_blas_factorizations == 0 stayed byte-identical to built-in dense.
        # n_factorizations is the total (= linear-solver setups), so the matrix
        # can show "<dgetrf>/<total>".
        "dense_blas_factorizations": int(stats.get("n_dense_blas_factorizations", 0)),
        "n_factorizations": int(stats.get("n_jac_evals", 0)),
        "cached": sim.codegen_cache_hit,
    }

    timing = {
        "io_sec": round(io_sec, 6),
        "parse_sec": round(libsbml_parse_sec, 6),
        "interpret_sec": round(interpret_sec, 6),
        "jac_derive_sec": round(jac_derive_sec, 6),
        "codegen_sec": round(codegen_sec, 6),
        # integrate_sec (= warm-min headline) + integrate_cold_sec / warm_min /
        # warm_max / warm_median / n_warm for the cold-vs-warm display.
        **integ,
        # Back-compat / cross-check: combined parse+interpret, what older reports
        # and the renderer's stock-RR fallback expect.
        "parse_interpret_sec": round(libsbml_parse_sec + interpret_sec, 6),
        "config": config,
        # The verdict run's solver work (steps, RHS/Jacobian evaluations): the
        # noise-free efficiency signal the nightly check compares (GH #702).
        "work": work_counters(stats),
    }

    return np.asarray(r.time), np.asarray(r.species), list(r.species_names), timing


def bn_ssa_replicates(
    xml: str,
    t_start: float,
    t_end: float,
    n_points: int,
    n_rep: int,
    seed_base: int,
    rep_timeout: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], dict, dict]:
    """``n_rep`` bngsim SSA replicates with GH #110 boundary diagnostics + timing.

    Like :func:`bn_ssa` but also aggregates each replicate's SSA boundary
    diagnostics over the ensemble — ``n_reverse_fires`` / ``n_negative_crossings``
    summed, plus the first non-empty offending reaction / species — and silences
    the per-replicate ``SsaBoundaryWarning`` (otherwise N× noise on a
    sign-indefinite model; the signal is surfaced once, structurally, via the
    returned ``ssa_diag``). ``n_reverse_fires > 0`` means a rate law evaluated
    **negative** somewhere on the trajectory and bngsim ran that channel backward
    with propensity ``|rate|`` (mean-faithful to the ODE) — i.e. the model is
    *runtime* sign-indefinite (GH #109), which the static t0 corpus gate cannot
    see. Returns ``(time, reps[n_rep,n_time,n_sp], names, ssa_diag, timing)``.

    The model is loaded once and cloned+reset per replicate; seeds run
    ``seed_base, seed_base+1, …`` so a comparison against RoadRunner uses an
    identical seed schedule. A model that fails ``validate_for_ssa`` raises
    ``bngsim.SsaValidationError`` (default strict_ssa=True) — the caller maps that
    to UNSUPPORTED, not EXCEPTION.

    The ``timing`` dict is measurement-only (#135): it never affects the returned
    trajectories. It carries the per-model SSA load (``parse_sec``/``interpret_sec``
    plus ``codegen_sec`` — the one-time structure-specialized propensity .so compile
    for eligible models, GH #190) read from the engine's public accessors, and the
    per-replicate ensemble simulation stats (:func:`_ensemble_stats` — ``sim.run``
    only, the fair per-trajectory analog of RoadRunner's per-replicate
    ``simulate``). The Simulator is built ONCE and reset+reseeded per replicate,
    matching RoadRunner's reuse-one-model pattern, so there is no per-replicate
    construction overhead (the prior clone+construct-per-replicate was both
    unrealistic and produced a contention-dominated "setup per rep" figure).
    """
    import statistics as _st
    import warnings

    import bngsim

    model = bngsim.Model.from_sbml(xml)

    # GH #190 — SSA propensity codegen. The structure-specialized propensity
    # vector is compiled to a content-cached .so (cc -O3) for eligible mass-action
    # / MichaelisMenten models and drives the RoadRunner-parity recompute-all
    # loop; ineligible models (Functional/Hill/Sat, events past the gate) stay on
    # the interpreted path. Compile (or cache-hit) it once here, timed, so the
    # per-model load tier reflects the real one-time codegen cost (the Simulator
    # constructs below reuse the cached .so). The authoritative backend actually
    # used is read back from the run's ssa_diagnostics below.
    ssa_codegen_sec = 0.0
    try:
        from bngsim._codegen import prepare_ssa_propensity_lib

        # force_recompile so the load tier shows the real one-time compile (fair
        # vs RoadRunner's cold per-process codegen), not a disk-cache hit from a
        # warm-up run earlier in this worker.
        _t_cg = time.perf_counter()
        _ssa_lib = prepare_ssa_propensity_lib(model, force_recompile=True)
        if _ssa_lib:
            ssa_codegen_sec = time.perf_counter() - _t_cg
    except Exception:
        ssa_codegen_sec = 0.0

    out: list[np.ndarray] = []
    names: list[str] = []
    times: np.ndarray | None = None
    ssa_diag = {
        "n_reverse_fires": 0,
        "first_reverse_reaction": "",
        "n_negative_crossings": 0,
        "first_negative_species": "",
    }
    run_secs: list[float] = []
    event_counts: list[int] = []
    prop_backend = "interpreted"  # GH #190 — overwritten from the run's diagnostics
    # Build the Simulator ONCE and reset+reseed per replicate, matching rr_ssa's
    # reuse-one-model pattern (RoadRunner builds once, then reset()+simulate() per
    # rep). Verified bit-identical to clone+reconstruct per replicate (reset()
    # restores the initial state and each run() seeds a fresh RNG from its `seed`
    # arg), so this is a fairness/realism fix only: a real SSA ensemble reuses one
    # Simulator, and the old per-replicate clone+construct (a) is not how ensembles
    # run and (b) produced a "setup per replicate" number that, timed under the
    # parallel screen, was dominated by CPU-contention noise (≈0.3 ms in isolation
    # ballooning to many ms), confusingly exceeding the per-replicate trajectory
    # time — while RoadRunner, reusing one model, reported no such overhead.
    sim = bngsim.Simulator(model, method="ssa")
    for rep in range(n_rep):
        model.reset()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", bngsim.SsaBoundaryWarning)
            t_run = time.perf_counter()
            # rep_timeout bounds each trajectory (the rr_parity adjudicator passes one
            # so a seed-dependent stochastic blowup — a runtime sign-indefinite rate
            # explosion that fires unboundedly on some seeds — raises SimulationTimeout
            # here and propagates, instead of hanging the whole ensemble). None = the
            # engine default (golden/other callers, unchanged).
            r = sim.run(
                t_span=(t_start, t_end),
                n_points=n_points,
                seed=seed_base + rep,
                timeout=rep_timeout,
            )
            run_secs.append(time.perf_counter() - t_run)
        if times is None:
            times = np.asarray(r.time)
            names = list(r.species_names)
        out.append(np.asarray(r.species))
        # SSA reaction-firing count for this trajectory: solver_stats["n_steps"] is
        # the number of Gillespie steps = reaction events fired. Summed/averaged
        # over the ensemble it gives the model's stochastic ACTIVITY (events per
        # unit simulated time) — the variable that actually drives per-trajectory
        # cost (far more than the species count does), so the matrix sorts and
        # plots on it.
        event_counts.append(int((r.solver_stats or {}).get("n_steps", 0) or 0))
        d = r.ssa_diagnostics
        ssa_diag["n_reverse_fires"] += int(d.get("n_reverse_fires", 0))
        ssa_diag["n_negative_crossings"] += int(d.get("n_negative_crossings", 0))
        if not ssa_diag["first_reverse_reaction"] and d.get("first_reverse_reaction"):
            ssa_diag["first_reverse_reaction"] = d["first_reverse_reaction"]
        if not ssa_diag["first_negative_species"] and d.get("first_negative_species"):
            ssa_diag["first_negative_species"] = d["first_negative_species"]
        # GH #190 — authoritative propensity backend the run actually used
        # ("cc"/"mir" = compiled recompute-all, "interpreted" = compute_propensity);
        # identical across replicates of one model, so the last value stands.
        prop_backend = d.get("propensity_backend", "interpreted")

    parse_sec = float(model.last_libsbml_parse_sec)
    interpret_sec = float(model.last_interpret_sec)
    span = max(float(t_end) - float(t_start), 1e-12)
    mean_events = _st.mean(event_counts) if event_counts else 0.0
    # Codegen only contributes to the load tier when the run actually used a
    # compiled kernel; an interpreted run pays no codegen even if a .so happened
    # to be on disk. "cached" distinguishes a fresh compile from a disk-cache hit.
    used_codegen = prop_backend in ("cc", "mir")
    codegen_sec = ssa_codegen_sec if used_codegen else 0.0
    timing = {
        # Per-model SSA load. The path-based loader folds the file read into the
        # libSBML parse, so io is not separately timed (io_sec=0); parse_sec
        # therefore includes the file read.
        "io_sec": 0.0,
        "parse_sec": round(parse_sec, 6),
        "interpret_sec": round(interpret_sec, 6),
        # GH #190 — one-time structure-spec propensity .so compile (0 when
        # interpreted or a disk-cache hit). Folded into the build subtotal so the
        # bngsim load tier is comparable to RoadRunner's parse+interpret+codegen+jit.
        "codegen_sec": round(codegen_sec, 6),
        "load_sec": round(parse_sec + interpret_sec + codegen_sec, 6),
        # Per-replicate ensemble simulation timing (sim.run only). The Simulator is
        # built once and reused across replicates (matching RoadRunner), so there is
        # no per-replicate setup overhead to report — the one-time construct is part
        # of load, and codegen is the codegen_sec line above.
        **_ensemble_stats(run_secs),
        # Stochastic activity: mean reaction events per trajectory and per unit
        # simulated time, averaged over the ensemble. The cost-driving axis.
        "events_per_rep": round(mean_events, 3),
        "events_per_time": round(mean_events / span, 4),
        "events_total": int(sum(event_counts)),
        # `codegen` is the actual propensity backend (matches the ODE config's
        # `codegen` key so the matrix renders it via rhs_backend_display);
        # `cached` flags a disk-cache hit vs a fresh compile.
        "config": {
            "method": "Gillespie SSA (exact)",
            "codegen": prop_backend,
            "cached": (codegen_sec < 0.005) if used_codegen else None,
        },
    }
    return times, np.stack(out, axis=0), names, ssa_diag, timing


def bn_ssa(
    xml: str, t_start: float, t_end: float, n_points: int, n_rep: int, seed_base: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """``n_rep`` bngsim SSA replicates. Returns (time, reps[n_rep,n_time,n_sp], names).

    Thin wrapper over :func:`bn_ssa_replicates` that drops the GH #110 boundary
    diagnostics and the #135 timing — the same engine call and seed schedule, for
    callers that don't need the sign-indefinite signal.
    """
    times, reps, names, _diag, _timing = bn_ssa_replicates(
        xml, t_start, t_end, n_points, n_rep, seed_base
    )
    return times, reps, names


# --------------------------------------------------------------------------- #
# RoadRunner adapters
# --------------------------------------------------------------------------- #
def _rr_colnames(res) -> list[str]:
    """Species names from an RR result: drop the time column, strip ``[…]``."""
    cols = [c[1:-1] if c.startswith("[") else c for c in res.colnames]
    return cols[1:]


# ── Event instants: RoadRunner sees a trigger only at its output times ───────
#
# RoadRunner does not fire a `time == c` event trigger between its output
# times, nor a time window such as `(time > a) && (time < b)` narrower than its
# integrator's step (strict or closed edges alike), unless an output time lands
# on the instant or inside the window. On a sweep grid with no sample there it
# skips the event, while bngsim (since #1015) stops on the instant and fires it,
# so the comparison reads a RoadRunner miss as a bngsim divergence. 10 BioModels
# were recorded as known artifacts for that (#1021). The instants a trigger's
# time-only comparisons change value at are found here from the SBML, so rr_ode
# can run RoadRunner through them and still report the job's own grid.

# A comparison more instants than this is not a schedule; leave it as it was.
_MAX_INSTANTS_PER_ATOM = 20_000


def _effectively_constant(model) -> set[str]:
    """Parameter and compartment ids nothing changes after the start: no rule
    targets them and no event assigns them. Their value at the start (after any
    initialAssignment) is their value throughout."""
    changed = {r.getVariable() for r in model.getListOfRules() if r.isSetVariable()}
    for e in model.getListOfEvents():
        changed.update(a.getVariable() for a in e.getListOfEventAssignments())

    # An algebraic rule names no variable; whatever non-constant symbol it reads
    # may be the one it determines.
    def names(node):
        yield node.getName()
        for i in range(node.getNumChildren()):
            yield from names(node.getChild(i))

    for r in model.getListOfRules():
        if r.isAlgebraic() and r.isSetMath():
            for name in names(r.getMath()):
                el = model.getParameter(name) if name else None
                el = el if el is not None else (model.getCompartment(name) if name else None)
                if el is not None and not el.getConstant():
                    changed.add(name)
    ids = [x.getId() for x in model.getListOfParameters()]
    ids += [x.getId() for x in model.getListOfCompartments()]
    return {i for i in ids if i not in changed}


def _mod_form(node, value_of):
    """``("rem", a, b, P)`` where ``node`` is a modulo of ``a*time + b`` by a
    constant P written out as ``x - P*floor(x/P)`` (or ``ceil``), or as the
    piecewise of the two that COPASI exports for ``%`` (libsbml prints it back as
    ``%``, but the AST holds the expansion)."""
    import libsbml

    ty = node.getType()
    if ty == libsbml.AST_FUNCTION_PIECEWISE:
        n = node.getNumChildren()
        forms = [_mod_form(node.getChild(i), value_of) for i in range(0, n, 2)]
        if forms and forms[0] is not None and all(f == forms[0] for f in forms):
            return forms[0]
        return None
    if ty != libsbml.AST_MINUS or node.getNumChildren() != 2:
        return None
    x, prod = node.getChild(0), node.getChild(1)
    if prod.getType() != libsbml.AST_TIMES or prod.getNumChildren() != 2:
        return None
    for p_node, f_node in (
        (prod.getChild(0), prod.getChild(1)),
        (prod.getChild(1), prod.getChild(0)),
    ):
        if f_node.getType() not in (libsbml.AST_FUNCTION_FLOOR, libsbml.AST_FUNCTION_CEILING):
            continue
        q = f_node.getChild(0) if f_node.getNumChildren() == 1 else None
        if q is None or q.getType() != libsbml.AST_DIVIDE or q.getNumChildren() != 2:
            continue
        arg, arg2 = _affine(x, value_of), _affine(q.getChild(0), value_of)
        per, per2 = _affine(p_node, value_of), _affine(q.getChild(1), value_of)
        if None in (arg, arg2, per, per2) or arg != arg2 or per != per2:
            continue
        if arg[0] == per[0] == "aff" and per[1] == 0.0 and per[2] != 0.0:
            return ("rem", arg[1], arg[2], per[2])
    return None


def _affine(node, value_of):
    """``node`` as ``("aff", a, b)`` meaning ``a*time + b``, as ``("rem", a, b, P)``
    meaning ``rem(a*time + b, P)``, or None where it is neither (a species, a
    parameter something changes, a function this does not read)."""
    import libsbml

    ty = node.getType()
    kids = [node.getChild(i) for i in range(node.getNumChildren())]
    if ty == libsbml.AST_NAME_TIME:
        return ("aff", 1.0, 0.0)
    if ty in (libsbml.AST_INTEGER, libsbml.AST_REAL, libsbml.AST_REAL_E, libsbml.AST_RATIONAL):
        return ("aff", 0.0, float(node.getValue()))
    if ty == libsbml.AST_CONSTANT_PI:
        return ("aff", 0.0, float(np.pi))
    if ty == libsbml.AST_CONSTANT_E:
        return ("aff", 0.0, float(np.e))
    if ty == libsbml.AST_NAME:
        v = value_of(node.getName())
        return None if v is None else ("aff", 0.0, float(v))
    if ty in (libsbml.AST_FUNCTION_PIECEWISE, libsbml.AST_MINUS):
        mod = _mod_form(node, value_of)
        if mod is not None:
            return mod
    parts = [_affine(k, value_of) for k in kids]
    if any(x is None for x in parts):
        return None
    if ty == libsbml.AST_FUNCTION_REM and len(parts) == 2:
        (x, p) = parts
        if x[0] == "aff" and p[0] == "aff" and p[1] == 0.0 and p[2] != 0.0:
            return ("rem", x[1], x[2], p[2])
        return None
    if any(x[0] != "aff" for x in parts):
        return None
    if ty == libsbml.AST_PLUS:
        return ("aff", sum(x[1] for x in parts), sum(x[2] for x in parts))
    if ty == libsbml.AST_MINUS:
        if len(parts) == 1:
            return ("aff", -parts[0][1], -parts[0][2])
        if len(parts) == 2:
            return ("aff", parts[0][1] - parts[1][1], parts[0][2] - parts[1][2])
        return None
    if ty == libsbml.AST_TIMES:
        a, b = 0.0, 1.0
        for x in parts:
            if x[1] != 0.0 and a != 0.0:
                return None  # time * time is not affine
            a, b = a * x[2] + x[1] * b, b * x[2]
        return ("aff", a, b)
    if ty == libsbml.AST_DIVIDE and len(parts) == 2 and parts[1][1] == 0.0 and parts[1][2] != 0.0:
        return ("aff", parts[0][1] / parts[1][2], parts[0][2] / parts[1][2])
    return None


def _atom_instants(left, right, t_start: float, t_end: float) -> tuple[list[float], list[float]]:
    """Where ``left ⋈ right`` can change value: ``(levels, wraps)``, the times
    the two sides meet and, for a ``rem``, the times it wraps back to 0."""
    if left is None or right is None:
        return [], []
    if left[0] == "rem" and right[0] == "aff" and right[1] == 0.0:
        left, right = right, left
    if left[0] == "aff" and left[1] == 0.0 and right[0] == "rem":
        # rem(a*t + b, P) against a constant r: where the argument passes r + kP
        # (the comparison's level) and kP (where rem wraps).
        r, (_, a, b, period) = left[2], right
        if a == 0.0:
            return [], []
        lo, hi = sorted((a * t_start + b, a * t_end + b))
        period = abs(period)
        if (hi - lo) / period > _MAX_INSTANTS_PER_ATOM / 2:
            return [], []
        out: tuple[list[float], list[float]] = ([], [])
        # rem takes values in (-P, P) (C's sign, or COPASI's floor form in [0, P)),
        # so a level outside that is never met.
        reachable = abs(r) < period and not (lo >= 0 and r < 0) and not (hi <= 0 and r > 0)
        levels = [r] if reachable else []
        for level, into in [*((lv, out[0]) for lv in levels), (0.0, out[1])]:
            k = np.arange(np.floor((lo - level) / period), np.ceil((hi - level) / period) + 1)
            into.extend(((level + k * period) - b) / a)
        return out
    if left[0] == "aff" and right[0] == "aff":
        a, b = left[1] - right[1], left[2] - right[2]
        return ([], []) if a == 0.0 else ([-b / a], [])
    return [], []


def event_instants(sbml: str, t_start: float, t_end: float, value_of=None) -> list[float]:
    """Output times in (t_start, t_end] that let RoadRunner see the time-only
    comparisons of the event triggers it would otherwise skip:

    * ``==``: the instant itself; RoadRunner does not fire it between samples.
      ``!=`` also gets a sample just after it, where it turns true again.
    * ``>``, ``>=``, ``<``, ``<=``: a sample just after each boundary (and, for
      a ``rem``, just after each wrap), which lands inside any window the
      comparison opens. RoadRunner fires a lone inequality at its own root, but
      not a window narrower than its integrator's step, strict or closed.
      Never a sample ON an inequality's boundary: one on the instant of a jump
      makes RoadRunner's integrator fail at the harness tolerances
      (MODEL0911120000's ``time >= 16.16`` pulse end: CV_CONV_FAILURE).

    A window that opens exactly at ``t_start`` gets no sample (its boundary is
    not inside the run); no corpus job has one.

    Reads comparisons of `time` (and affine forms of it, `rem` of one included)
    against constants: numbers, and parameters or compartments nothing changes
    after the start, valued by ``value_of(id)`` (RoadRunner's value after the
    initial assignments; the SBML value when None). A comparison that reads a
    species, or anything else, gives no instants: RoadRunner finds those itself.
    """
    import libsbml

    doc = libsbml.readSBMLFromString(sbml)
    model = doc.getModel()
    if model is None or model.getNumEvents() == 0:
        return []
    constant = _effectively_constant(model)
    if value_of is None:

        def value_of(i):
            par = model.getParameter(i)
            if par is not None:
                return par.getValue() if par.isSetValue() else None
            comp = model.getCompartment(i)
            return comp.getSize() if comp is not None and comp.isSetSize() else None

    def known(i):
        return value_of(i) if i in constant else None

    relational = {
        libsbml.AST_RELATIONAL_EQ,
        libsbml.AST_RELATIONAL_NEQ,
        libsbml.AST_RELATIONAL_GT,
        libsbml.AST_RELATIONAL_GEQ,
        libsbml.AST_RELATIONAL_LT,
        libsbml.AST_RELATIONAL_LEQ,
    }
    equality = {libsbml.AST_RELATIONAL_EQ, libsbml.AST_RELATIONAL_NEQ}
    out: set[float] = set()

    def add(boundary: float, after: bool = False) -> None:
        # The boundary decides whether it is inside the run, not the sample.
        if not (np.isfinite(boundary) and t_start < boundary <= t_end):
            return
        t = boundary + 1e-9 * max(1.0, abs(boundary)) if after else boundary
        if t <= t_end:
            out.add(float(t))

    def walk(node):
        ty = node.getType()
        if ty in relational:
            sides = [_affine(node.getChild(i), known) for i in range(node.getNumChildren())]
            for left, right in zip(sides, sides[1:], strict=False):
                levels, wraps = _atom_instants(left, right, t_start, t_end)
                if ty in equality:
                    for t in levels:
                        add(t)
                        if ty == libsbml.AST_RELATIONAL_NEQ:
                            add(t, after=True)
                else:
                    for t in levels + wraps:
                        add(t, after=True)
            return
        for i in range(node.getNumChildren()):
            walk(node.getChild(i))

    for e in model.getListOfEvents():
        if e.isSetTrigger() and e.getTrigger().isSetMath():
            walk(e.getTrigger().getMath())
    return sorted(out)


def rr_ode(
    xml: str,
    t_start: float,
    t_end: float,
    n_points: int,
    rtol: float,
    atol: float,
    *,
    through_event_instants: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """One RoadRunner cvode run. Returns (time, values[n_time,n_sp], names, timing).

    With ``through_event_instants`` RoadRunner is also given output times at the
    instants :func:`event_instants` finds, so it fires the events a sample on
    the job grid would miss, and the rows on the job grid are what is returned.
    A model whose triggers give no instants runs exactly as without it. The
    timing dict records ``event_instants`` (how many were added).

    Selections are forced to **concentration** (``[id]``) for every species,
    floating and boundary alike. RoadRunner's *default* timecourse selection
    reports floating species as concentration (``[X]``) but **boundary** species
    as *amount* (the bare ``X`` symbol). For a boundary species in a V≠1
    compartment that silently makes the default output amount-vs-concentration
    against bngsim (which always reports concentration), manufacturing a spurious
    1/V "divergence" — e.g. BIOMD0000000659, all-boundary rate-rule model, nucleus
    species off by exactly V_n=0.5. Requesting ``[id]`` for all species keeps the
    comparison concentration-vs-concentration as the suite documents (floating
    species are unchanged; ``[X]`` is identical to their default).

    RR's default selection also *omits boundary species entirely* for some
    models: one with 0 floating species whose only dynamic species are
    boundary (BIOMD0000000567 — ``A`` constant, ``B`` an assignmentRule target,
    both ``boundaryCondition=true`` ⇒ default selection is just ``['time']``).
    Those would leave the bngsim/RR species sets disjoint even though both
    engines compute (and agree on) the boundary trajectories. We add any
    boundary species missing from the default as ``[id]`` concentration so they
    enter the comparison instead.

    The timing dict carries the per-phase load breakdown. With the instrumented
    RoadRunner build (#135, ``getLoadTimings()``) it is split into
    ``read_sec``/``parse_sec``/``interpret_sec``/``codegen_sec``/``jit_sec`` plus
    ``model_cache_hit``; on stock 2.9.2 it degrades to the single lumped
    ``parse_interpret_codegen_sec`` (the matrix renderer prefers the split and
    ignores the lumped value when ``parse_sec`` is present). ``io_sec`` (Python
    file read) and ``integrate_sec`` (``rr.simulate``) are always timed here.
    """
    from pathlib import Path

    import roadrunner

    # 1. File I/O (if xml is a path; SBML text itself can be too long for a
    # file name, which Path.exists() raises on)
    t0 = time.perf_counter()
    if not str(xml).lstrip().startswith("<") and Path(xml).exists():
        xml_string = Path(xml).read_text()
        io_sec = time.perf_counter() - t0
    else:
        # Already a string
        xml_string = xml
        io_sec = 0.0

    # 2. Parse + Interpret + LLVM JIT: XML string -> RoadRunner model ready to simulate
    # (libSBML parse + C++ interpretation + LLVM JIT compilation, inseparable)
    t0 = time.perf_counter()
    rr = roadrunner.RoadRunner(xml_string)
    rr.integrator = "cvode"
    rr.integrator.relative_tolerance = rtol
    rr.integrator.absolute_tolerance = atol
    # Start from RR's own default timecourse selection (which already includes
    # rate-rule-target parameters that no-species models expose, e.g.
    # MODEL0406553884 with 0 floating/boundary species) and only rewrite the
    # bare-id *boundary* species columns — those report amount — to ``[id]``
    # concentration. Floating species are already ``[X]`` and stay untouched.
    boundary = list(rr.model.getBoundarySpeciesIds())
    boundary_set = set(boundary)
    sel = [f"[{s}]" if s in boundary_set else s for s in rr.timeCourseSelections]
    # Append boundary species the default selection dropped entirely, so a
    # 0-floating-species model's boundary trajectories still get compared.
    present = {s[1:-1] if s.startswith("[") else s for s in sel}
    sel.extend(f"[{s}]" for s in boundary if s not in present)
    rr.timeCourseSelections = sel
    parse_interpret_codegen_sec = time.perf_counter() - t0

    # Per-phase load breakdown from the instrumented RoadRunner build (#135).
    # Stock 2.9.2 has no getLoadTimings() → fall back to the lumped wall above.
    try:
        load = dict(rr.getLoadTimings())
    except AttributeError:
        load = {}

    instants: list[float] = []
    if through_event_instants:

        def _value(i):
            try:
                return float(rr.getValue(i))
            except Exception:
                return None

        instants = event_instants(xml_string, t_start, t_end, _value)
    grid = np.linspace(t_start, t_end, n_points) if instants else None
    times = list(np.union1d(grid, instants)) if instants else None

    def _simulate():
        if times is None:
            return rr.simulate(t_start, t_end, n_points)
        return rr.simulate(times=times)

    # 3. Integration (cold + warm), same recipe as bngsim for a fair comparison.
    # First simulate() is COLD (RR's one-time CVODE first-solve setup); its
    # trajectory feeds the parity verdict. Warm reps reset() to IC each (RR
    # continues from state otherwise) and reuse the warm integrator.
    t1 = time.perf_counter()
    res = _simulate()
    cold_sec = time.perf_counter() - t1
    warm: list[float] = []
    for _ in range(_warm_rep_count(cold_sec)):
        # Best-effort timing only — never let a warm re-solve change the verdict.
        # RoadRunner can hit CV_TOO_MUCH_WORK on a re-integration of a stiff model
        # whose cold solve succeeded (e.g. BIOMD338/339); swallow and stop.
        try:
            rr.reset()
            t1 = time.perf_counter()
            _simulate()
            warm.append(time.perf_counter() - t1)
        except Exception:
            break
    integ = _integrate_stats(cold_sec, warm)

    # Build timing dict
    timing = {
        "io_sec": round(io_sec, 6),
        # Cross-check / stock-RR fallback; the renderer prefers the split keys
        # below and ignores this whenever parse_sec is present.
        "parse_interpret_codegen_sec": round(parse_interpret_codegen_sec, 6),
        # integrate_sec (= warm-min headline) + cold/min/max/median/n_warm.
        **integ,
        "event_instants": len(instants),
        "config": {
            "codegen": "LLVM JIT",
            # CVODE uses a difference-quotient Jacobian (CVODEIntegrator.cpp:745,
            # Jac=nullptr) and the built-in SUNDIALS dense LU (:733); neither is
            # configurable — see the #135 method-detection correction.
            "jacobian": "FD (difference-quotient)",
            "linear_solver": "Dense (built-in LU)",
            "cached": bool(load.get("model_cache_hit", 0.0)),
        },
    }
    if load:
        # The matrix's phase model has a single "codegen" slot and no JIT column,
        # so fold LLVM JIT into codegen (codegen_sec = IR-gen + JIT) to keep the
        # RoadRunner total honest — jit_sec dominates the LLVM load. The raw IR /
        # JIT split is preserved under codegen_ir_sec / jit_sec for provenance.
        ir = load.get("codegen_sec", 0.0)
        jit = load.get("jit_sec", 0.0)
        timing.update(
            {
                "read_sec": round(load.get("read_sec", 0.0), 6),
                "parse_sec": round(load.get("parse_sec", 0.0), 6),
                "interpret_sec": round(load.get("interpret_sec", 0.0), 6),
                "codegen_sec": round(ir + jit, 6),
                "codegen_ir_sec": round(ir, 6),
                "jit_sec": round(jit, 6),
                "model_cache_hit": load.get("model_cache_hit", 0.0),
            }
        )

    names = _rr_colnames(res)
    arr = np.asarray(res)
    if times is not None:
        rows = np.searchsorted(arr[:, 0], grid)
        if rows.max() >= len(arr) or not np.allclose(arr[rows, 0], grid, rtol=0, atol=1e-9):
            raise RuntimeError("RoadRunner did not report every time of the job grid")
        arr = arr[rows]
    return arr[:, 0].copy(), arr[:, 1:], names, timing


def rr_ssa(
    xml: str, t_start: float, t_end: float, n_points: int, n_rep: int, seed_base: int
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """``n_rep`` RoadRunner gillespie replicates on the bngsim seed schedule.

    Returns ``(time, reps[n_rep,n_time,n_sp], names, timing)``. The ``timing`` dict
    is measurement-only (#135): the per-model load breakdown from the instrumented
    RoadRunner build (``getLoadTimings()`` — read/parse/interpret/codegen/jit plus
    ``model_cache_hit``; the load phases are integrator-agnostic, so they apply to
    the gillespie path exactly as on the ODE side) and the per-replicate ensemble
    integration stats (:func:`_ensemble_stats` over each ``simulate`` call). Stock
    RoadRunner without ``getLoadTimings()`` degrades to ``load={}`` (the load split
    is simply absent). The model is built once; each replicate resets, reseeds,
    and re-simulates.
    """
    rr = _rr_gillespie(xml)
    # Per-phase load breakdown from the instrumented RoadRunner build (#135);
    # absent on stock 2.9.2 (no getLoadTimings()).
    try:
        load = dict(rr.getLoadTimings())
    except AttributeError:
        load = {}
    out: list[np.ndarray] = []
    names: list[str] = []
    times: np.ndarray | None = None
    rep_secs: list[float] = []
    for rep in range(n_rep):
        _rr_reseed(rr, seed_base + rep)
        t_run = time.perf_counter()
        res = rr.simulate(t_start, t_end, n_points)
        rep_secs.append(time.perf_counter() - t_run)
        arr = np.asarray(res)
        if times is None:
            times = arr[:, 0].copy()
            names = _rr_colnames(res)
        out.append(arr[:, 1:])

    ir = load.get("codegen_sec", 0.0)
    jit = load.get("jit_sec", 0.0)
    timing = {
        "io_sec": round(load.get("read_sec", 0.0), 6),
        "parse_sec": round(load.get("parse_sec", 0.0), 6),
        "interpret_sec": round(load.get("interpret_sec", 0.0), 6),
        # Fold LLVM IR-gen + JIT into one "codegen" slot (the matrix has no
        # separate JIT column), keeping the raw split for provenance.
        "codegen_sec": round(ir + jit, 6),
        "codegen_ir_sec": round(ir, 6),
        "jit_sec": round(jit, 6),
        "load_sec": round(
            load.get("read_sec", 0.0)
            + load.get("parse_sec", 0.0)
            + load.get("interpret_sec", 0.0)
            + ir
            + jit,
            6,
        ),
        "model_cache_hit": load.get("model_cache_hit", 0.0),
        **_ensemble_stats(rep_secs),
        "config": {
            "method": "Gillespie SSA (exact)",
            "codegen": "LLVM JIT",
            "cached": bool(load.get("model_cache_hit", 0.0)),
        },
    }
    return times, np.stack(out, axis=0), names, timing


def _rr_gillespie(xml: str):
    """RoadRunner's model of ``xml`` on its gillespie integrator at a fixed step,
    as :func:`rr_ssa` and :func:`rr_ssa_nonfinite_rate` both run it."""
    import roadrunner

    rr = roadrunner.RoadRunner(xml)
    rr.integrator = "gillespie"
    rr.integrator.variable_step_size = False
    return rr


def _rr_reseed(rr, seed: int) -> None:
    """Start a replicate: the model back at its initial state, on ``seed``."""
    rr.reset()
    rr.integrator.seed = int(seed)


def rr_ssa_nonfinite_rate(
    xml: str, t_start: float, t_end: float, n_points: int, n_rep: int, seed_base: int
) -> dict:
    """Whether RoadRunner's gillespie, on the bngsim seed schedule, reaches a state
    where one of its own reaction rates is NaN or infinite (issue #1211).

    RoadRunner raises nothing there: its gillespie fires no further reaction and
    holds that state to ``t_end`` (measured on 2.9.2). So a replicate that reached
    one ends on it, and the rates are read at the end of each replicate, which runs
    as in :func:`rr_ssa`. Stops at the first such replicate. Returns
    ``{"replicates", "seed", "boundary", "nonfinite"}``: the replicates run, the
    seed of the one that ended on a non-finite rate, the boundary species, and every
    reaction whose rate is non-finite there, as ``{"reaction", "rate", "net"}``: its
    id, its rate (``repr``, so the record stays JSON) and its net change of each
    floating species, from RoadRunner's stoichiometry matrix. ``seed`` is None and
    ``nonfinite`` empty when every rate stayed finite. Raises what RoadRunner
    raises, for example on a model with rate rules, which its gillespie refuses.
    """
    rr = _rr_gillespie(xml)
    reaction_ids = list(rr.model.getReactionIds())
    boundary = list(rr.model.getBoundarySpeciesIds())
    for rep in range(n_rep):
        _rr_reseed(rr, seed_base + rep)
        rr.simulate(t_start, t_end, n_points)
        rates = np.asarray(rr.model.getReactionRates(), dtype=float)
        bad = np.flatnonzero(~np.isfinite(rates))
        if bad.size:
            stoich = rr.getFullStoichiometryMatrix()
            rows, cols = list(stoich.rownames), list(stoich.colnames)
            m = np.asarray(stoich, dtype=float)
            nonfinite = []
            for j in bad:
                c = cols.index(reaction_ids[j])
                net = {rows[i]: float(m[i, c]) for i in range(len(rows)) if m[i, c] != 0.0}
                nonfinite.append(
                    {"reaction": reaction_ids[j], "rate": repr(float(rates[j])), "net": net}
                )
            return {
                "replicates": rep + 1,
                "seed": int(seed_base + rep),
                "boundary": boundary,
                "nonfinite": nonfinite,
            }
    return {"replicates": int(n_rep), "seed": None, "boundary": boundary, "nonfinite": []}


# --------------------------------------------------------------------------- #
# Process-isolation scheduler (kill-on-overrun)
# --------------------------------------------------------------------------- #
def schedule(
    specs: list[dict],
    target,
    *,
    workers: int,
    timeout_of,
    on_done=None,
) -> list[dict]:
    """Run ``target(spec, q)`` for each spec in up to ``workers`` subprocesses.

    Each spec is a plain (picklable) dict with a unique ``"key"``. ``target``
    is a module-level worker that puts exactly one result dict on the queue;
    ``timeout_of(spec)`` gives that spec's wall-clock cap in seconds. On
    overrun the child is terminated and a ``{"status": "timeout"}`` result is
    synthesized; a child that dies without a result yields
    ``{"status": "dead", "exitcode": …}``. ``on_done(finished, total, res)``
    is called for progress as each result lands. Results come back in spec
    order.

    Uses the 'spawn' start method so the C extensions (bngsim, RoadRunner) are
    confined to the disposable child — the only safe way to survive an engine
    segfault on a pathological model.

    The queue is drained *before* the child is joined: a child that puts a
    large payload (e.g. a full golden trajectory) can't exit until its feeder
    thread flushes that item to the pipe, which can't happen until the parent
    reads — so gating the read on the child already being dead would deadlock
    (the standard multiprocessing.Queue caveat). We poll ``get_nowait`` each
    tick, which keeps the pipe draining and lets the child finish.
    """
    ctx = mp.get_context("spawn")
    pending = list(specs)
    running: list[dict] = []
    done: dict[str, dict] = {}
    total = len(specs)
    finished = 0
    post_result_join_grace = 30.0

    def _finish(spec, res):
        nonlocal finished
        done[spec["key"]] = res
        finished += 1
        if on_done:
            on_done(finished, total, res)

    while pending or running:
        while pending and len(running) < workers:
            spec = pending.pop(0)
            q = ctx.Queue()
            proc = ctx.Process(target=target, args=(spec, q))
            proc.start()
            running.append(
                {"proc": proc, "q": q, "spec": spec, "start": time.time(), "cap": timeout_of(spec)}
            )

        still = []
        for r in running:
            proc, spec, q = r["proc"], r["spec"], r["q"]
            if "res" not in r:
                with contextlib.suppress(_queue.Empty):
                    r["res"] = q.get_nowait()  # drain FIRST so the child can flush/exit
            if "res" in r:
                # Keep the process handle live until the child really exits. Dropping
                # it after a bounded join can leave a spawned child orphaned after a
                # successful q.put(), which wedged long stochastic sweeps at shutdown.
                r.setdefault("result_time", time.time())
                proc.join(timeout=0.05)
                if proc.is_alive():
                    if time.time() - r["result_time"] > post_result_join_grace:
                        proc.terminate()
                        proc.join(timeout=5.0)
                        if proc.is_alive():
                            proc.kill()
                            proc.join()
                        _finish(spec, r["res"])
                    else:
                        still.append(r)
                else:
                    _finish(spec, r["res"])
            elif time.time() - r["start"] > r["cap"]:
                proc.terminate()
                proc.join()
                _finish(
                    spec,
                    {
                        **spec,
                        "status": "timeout",
                        "wall": round(time.time() - r["start"], 1),
                        "cap": r["cap"],
                    },
                )
            elif not proc.is_alive():
                proc.join()
                try:  # a result may have landed on the queue as the child exited
                    res = q.get(timeout=1.0)
                except _queue.Empty:
                    res = {**spec, "status": "dead", "exitcode": proc.exitcode}
                _finish(spec, res)
            else:
                still.append(r)
        running = still
        if running:
            time.sleep(0.05)

    return [done[s["key"]] for s in specs]


# --------------------------------------------------------------------------- #
# Job loading / filtering (shared by both entrypoints)
# --------------------------------------------------------------------------- #
def load_and_filter(jobs, args, *, suite_dir: Path):
    """Apply the shared ``--models/--include/--exclude/--limit/--effort`` filters.

    Returns the filtered Job list. ``args`` must carry those attributes
    (``--effort`` only prunes tiered SSA jobs; see ``effort_allows``).
    """
    out = list(jobs)
    if getattr(args, "models", ""):
        wanted = {x.strip() for x in args.models.split(",") if x.strip()}
        out = [j for j in out if j.model_id in wanted]
    if getattr(args, "include", ""):
        out = [j for j in out if args.include in j.model]
    if getattr(args, "exclude", ""):
        out = [j for j in out if args.exclude not in j.model]
    if getattr(args, "effort", None):
        out = [j for j in out if effort_allows(args.effort, j.params.get("effort"))]
    if getattr(args, "limit", 0):
        out = out[: args.limit]
    return out


def model_path(suite_dir: Path, job) -> Path:
    """Absolute path to a job's vendored SBML (``Job.model`` is suite-relative)."""
    return (suite_dir / job.model).resolve()
