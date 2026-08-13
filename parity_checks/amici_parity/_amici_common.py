"""AMICI reference adapter for the bngsim-vs-AMICI ODE parity suite.

This is the AMICI analogue of ``rr_parity/_rr_common.py``'s reference block
(``rr_ode``). The bngsim test side (``bn_ode``) and every engine-agnostic helper
(``_integrate_stats``, ``_warm_rep_count``, ``schedule``, ``hardware_info``,
``DEFAULT_RTOL``/``DEFAULT_ATOL``) are imported verbatim from
``rr_parity._rr_common`` so the two suites share ONE bngsim adapter and ONE
integration-timing taxonomy. This module adds only what is AMICI-specific:

  * ``amici_ode``            — the reference run (build / load / integrate), with a
                              timing dict mirroring ``rr_ode``'s schema.
  * ``align_common``         — the shared species aligner, wrapped to undo AMICI's
                              ``amici_`` id-collision prefix first (issue #321).
                              AMICI's naming rule, so it does not belong in the
                              generic helper RoadRunner also uses.
  * ``measure_warmup``       — per-process warmup (bngsim SymPy + AMICI import).
  * ``set_amici_quiet``      — silence AMICI's logger and compiler chatter.
  * ``classify_reference_refusal`` — map an AMICI failure to a refusal subclass.

AMICI 1.0.1 API notes (verified against the installed source; the older
``getModel``/``runAmiciSimulation`` names used by ``harness/comparison/
bench_ode_vs_amici.py`` are GONE):
    importer = amici.SbmlImporter(sbml_str, from_file=False)        # libSBML parse
    importer.sbml2amici(name, out_dir, generate_sensitivity_code=False,
                        compute_conservation_laws=False, verbose=False)  # C++ codegen+compile
    mod   = amici.import_model_module(name, out_dir)                # load the .so
    model = mod.get_model()
    solver = model.create_solver()
    solver.set_relative_tolerance(rtol); solver.set_absolute_tolerance(atol)
    solver.set_sensitivity_order(SensitivityOrder_none)
    model.set_timepoints(ts)
    rdata = model.simulate(solver=solver)                           # rdata.x/.ts/.state_ids/.status/.cpu_time

Unlike RoadRunner (whose ``RoadRunner(xml)`` fuses parse+interpret+JIT, forcing a
C++-instrumented build to split them), AMICI exposes every cost tier at the Python
level, so NO AMICI source patch / rebuild is needed:
    parse   = wall around SbmlImporter(...)         (libSBML parse + symbolic ingest)
    codegen = wall around sbml2amici(...)           (symbolic analytic Jacobian +
                                                     C++ generation + gcc/clang compile);
                                                     0 on a disk-cache hit
    load    = wall around import_model_module(...)  (load the compiled extension)
    integrate = cold + warm model.simulate() reps   (the shared cold/warm taxonomy)
``rdata.cpu_time`` is the pure CVODES integration time in MILLISECONDS (kept under
``integrate_cpu_ms`` as a cross-check against the Python wall ``integrate_*_sec``).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Reuse the bngsim adapter + engine-agnostic helpers from rr_parity. _rr_common
# has NO module-level ``import roadrunner`` (it is imported lazily inside rr_ode /
# _rr_warmup only), so importing it here never pulls RoadRunner into an AMICI run.
# --------------------------------------------------------------------------- #
_PARITY_CHECKS = Path(__file__).resolve().parent.parent
if str(_PARITY_CHECKS) not in sys.path:
    sys.path.insert(0, str(_PARITY_CHECKS))

from rr_parity import _rr_common as _rc  # noqa: E402

# Re-export the shared surface so amici_run.py imports only this module.
bn_ode = _rc.bn_ode
schedule = _rc.schedule
hardware_info = _rc.hardware_info
load_and_filter = _rc.load_and_filter
model_path = _rc.model_path
sundials_version = _rc.sundials_version
_integrate_stats = _rc._integrate_stats
_warm_rep_count = _rc._warm_rep_count
DEFAULT_RTOL = _rc.DEFAULT_RTOL
DEFAULT_ATOL = _rc.DEFAULT_ATOL


# --------------------------------------------------------------------------- #
# AMICI's `amici_` id-collision prefix (issue #321)
# --------------------------------------------------------------------------- #
# AMICI renames any SBML id that collides with a symbol in its own generated C++
# namespace — `x` IS the state vector there — so `<species id="x">` is reported
# as `amici_x` in `rdata.state_ids`. The exact-id intersection in
# `_rr_common.align_common` then comes out EMPTY and the job is reported as a
# structural loader divergence at value=inf: the loudest verdict the suite has,
# on models where the two engines agree perfectly.
#
# This is the species-side twin of bngsim's `_lp_` local-parameter prefix, which
# `_amici_sens.bn_param_to_sbml_id` already strips — and it gets the same
# discipline: the prefix is POSITIONAL (only a leading occurrence), and an
# ambiguous strip is DROPPED rather than guessed at. Guessing is the dangerous
# failure here: mis-pairing two different species produces a confident-looking
# DIFF, which is far worse than the loud one it replaces.
#
# Kept in the AMICI adapter, not in `_rr_common`, because it is AMICI's naming
# rule — RoadRunner never emits it, and the shared helper should stay generic.
AMICI_ID_PREFIX = "amici_"


def amici_state_id(name: str) -> str:
    """Map one AMICI-reported state id back to its SBML id."""
    if name.startswith(AMICI_ID_PREFIX):
        return name[len(AMICI_ID_PREFIX) :]
    return name


def normalize_amici_names(am_names: list[str]) -> list[str]:
    """De-prefix AMICI's state ids, leaving any ambiguous one untouched.

    Ambiguity means two reported ids normalize to the same SBML id — e.g. a model
    that genuinely declares both ``x`` and ``amici_x``. Stripping there would
    silently pair the wrong series, so both are left as-is: they simply fail to
    match, which is a visible non-comparison rather than an invisible wrong one.
    """
    stripped = [amici_state_id(n) for n in am_names]
    counts: dict[str, int] = {}
    for s in stripped:
        counts[s] = counts.get(s, 0) + 1
    return [s if counts[s] == 1 else raw for raw, s in zip(am_names, stripped, strict=True)]


# --------------------------------------------------------------------------- #
# bngsim's DECLARED refusals (issues #319, #320, #323)
# --------------------------------------------------------------------------- #
# Recognized BY TYPE, never by message text. These refusals carry long prose that
# cites GH issue numbers and gets reworded; a prefix match would silently demote
# every one of them back to EXCEPTION the first time someone edits a sentence,
# which is the exact signal dilution UNSUPPORTED exists to remove.
#
#   SensitivityUnsupportedError — bngsim inspected the model and will not
#       differentiate this construct (a non-differentiable event crossing time,
#       a rate law with no closed-form derivative, a switch-time parameter that
#       also acts in-branch). Sensitivity job only.
#   UnderSpecifiedModelError    — the model reads a symbol it never defines, so
#       bngsim refuses to guess 0.0. Raised at LOAD, so it reaches both jobs.
#
# Both are clean, documented, non-scoring refusals — not actionable bngsim bugs.
_DECLARED_REFUSAL_NAMES = ("SensitivityUnsupportedError", "UnderSpecifiedModelError")


def is_declared_refusal(exc: BaseException) -> bool:
    """True when bngsim DECLARED it will not handle this model.

    ``getattr`` rather than a hard import so a bngsim predating either class
    degrades to the old EXCEPTION bucket instead of failing every job.
    """
    import bngsim

    types = tuple(
        t for t in (getattr(bngsim, n, None) for n in _DECLARED_REFUSAL_NAMES) if t is not None
    )
    return bool(types) and isinstance(exc, types)


def align_common(bn_names: list[str], am_names: list[str]):
    """``_rr_common.align_common`` with AMICI's ``amici_`` prefix undone first.

    Same ``(bn_idx, am_idx, common)`` contract, and ``common`` carries SBML ids
    (bngsim's namespace), so downstream comment strings stay readable. The
    returned ``am_idx`` still indexes the ORIGINAL ``am_names`` order, which is
    what the caller slices its ``rdata`` columns with.
    """
    return _rc.align_common(bn_names, normalize_amici_names(am_names))


# On-disk cache of compiled AMICI extensions, keyed by SBML content + build flags.
# AMICI does NO automatic staleness check, so the key is the whole hash: a changed
# model or build flag yields a fresh directory. Gitignored; the first sweep pays
# the cold C++ compile, every re-run is a load-only cache hit.
AMICI_CACHE = Path(__file__).resolve().parent / "amici_cache"


def ensure_build_path() -> str:
    """Put this environment's script dir at the FRONT of ``PATH`` and return it.

    AMICI compiles each model by shelling out to ``cmake``, which locates SWIG via
    ``find_package(SWIG)`` — a plain ``PATH`` search. The ``amici`` dependency
    group installs its own matching ``swig`` into ``.venv/bin/``, but a worker
    launched as ``.venv/bin/python …`` from a shell that never *activated* the venv
    does NOT have that directory on ``PATH``. Two failure modes follow, both of
    which cost real debugging time:

      * no swig anywhere ⇒ ``cmake`` fails with "Could NOT find SWIG
        (SWIG_EXECUTABLE SWIG_DIR)" and EVERY model lands in the REFERENCE_FAILED
        bucket with refusal=compile — a whole sweep of false negatives that looks
        like an AMICI feature gap rather than a missing tool;
      * a *system* swig (e.g. homebrew) is found first ⇒ it builds, but against a
        different SWIG than AMICI's own core was wrapped with, and AMICI warns
        "SWIG version mismatch … may lead to unexpected behavior" — i.e. a
        silently-suspect reference oracle, which is worse than a hard failure.

    Prepending (not appending) the script dir fixes both: the venv's pinned swig
    wins over any system copy. Idempotent — calling twice does not duplicate the
    entry.

    The directory comes from ``sysconfig.get_path("scripts")``, NOT from
    ``Path(sys.executable).parent``. In a uv-created venv ``.venv/bin/python`` is a
    symlink to an interpreter under ``~/.local/share/uv/python/…/bin``, so
    ``.resolve()`` lands in the *base* interpreter's bin — which has no swig — and
    the entry is silently useless. ``sysconfig`` reports the venv's own script dir
    and is correct on Windows (``Scripts``) too.
    """
    import sysconfig

    bindir = sysconfig.get_path("scripts")
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if not parts or parts[0] != bindir:
        os.environ["PATH"] = os.pathsep.join([bindir] + [p for p in parts if p != bindir])
    return bindir


# Build flags pinned for parity (NOT AMICI defaults):
#   generate_sensitivity_code=False — pure forward ODE; skips the sensitivity C++.
#   compute_conservation_laws=False — keep every species an independent STATE so the
#       AMICI state set matches what bngsim/RR emit (CL elimination would drop the
#       eliminated species from rdata.x and shrink the compared set). Coverage > the
#       small integration saving for a correctness suite.
#   observation_model=[] — no observables/likelihood; we compare the state vector
#       rdata.x, so the y/sigmay/Jy/dJydy C++ is dead weight. Skipping it makes the
#       build a true pure-ODE cost (≈halves compile on small models) and the phase
#       breakdown clean (no observable functions). v2 = invalidates the v1 cache.
_BUILD_FLAGS = "sens=0;cl=0;obs=0;v2"

# --------------------------------------------------------------------------- #
# Build-phase decomposition. AMICI self-times every build phase with its
# @log_execution_time decorator, logging "Finished {desc}{pad}{+recursion}
# ({dur:.2E}s)" — but pins its module loggers to ERROR and gates them behind the
# ``verbose`` arg, so the records are normally invisible. We lower those loggers to
# DEBUG and capture the records to decompose the otherwise-opaque sbml2amici() into
# parse / interpret / Jacobian-derivation / codegen / compile. NO AMICI C++ patch is
# needed: all the symbolic work is Python/SymPy (timed per equation) and the only
# C++ step — the compile — is itself a single timed phase ("compiling cpp code").
# Empirically the compile dominates (~95% of build on small models, a ~25-30s floor:
# cmake-configure + ninja over the SUNDIALS-interface boilerplate + SWIG + static
# link), while the analytic-Jacobian symbolic derivation is milliseconds (and grows
# with model size). See dev probe 2026-06-15.
# --------------------------------------------------------------------------- #
_LOG_RX = re.compile(r"^Finished (.+?)\s*\+*\s*\(([\d.eE+-]+)s\)\s*$")

# libSBML-level work → "parse".
_PARSE_PHASES = frozenset(
    {
        "loading SBML",
        "validating SBML",
        "re-validating SBML",
        "flattening hierarchical SBML",
        "converting SBML functions",
        "converting SBML local parameters",
    }
)
# The dxdot/dx symbolic chain AMICI differentiates for the analytic Jacobian (the
# forward-ODE Jacobian is assembled from these). Their "computing X"/"simplifying X"
# phases ARE the analytic-Jacobian derivation cost.
_JAC_FUNCS = ("dwdx", "dwdw", "dxdotdw", "dxdotdx_explicit", "dxdotdx", "JSparse", "JB", "J")

# AMICI emits these phases from two module loggers; lower them to DEBUG during a
# build so the records reach our capture handler.
_AMICI_BUILD_LOGGERS = ("amici.exporters.sundials.de_export", "amici.importers.sbml")


def _is_jac_func(fn: str) -> bool:
    return any(fn == j or fn.startswith(j) for j in _JAC_FUNCS)


@contextmanager
def _silence_fds():
    """Redirect OS-level stdout+stderr to /dev/null. AMICI's build shells out to
    cmake/ninja/swig (subprocess output on fd 1/2) and verbose=DEBUG would otherwise
    flood the sweep log; this mutes both the subprocess output and the logger's own
    console handler, while our capture handler still receives every record (it
    appends to a list, not a file descriptor)."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved_out, saved_err = os.dup(1), os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        for fd in (devnull, saved_out, saved_err):
            os.close(fd)


class _PhaseCapture(logging.Handler):
    """Collect (description, duration_seconds) from AMICI's log_execution_time records."""

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.phases: list[tuple[str, float]] = []

    def emit(self, record):
        m = _LOG_RX.match(record.getMessage())
        if m:
            self.phases.append((m.group(1).strip(), float(m.group(2))))


def _bucket_phases(phases) -> dict:
    """Bucket captured AMICI build phases into the parity taxonomy (seconds):
    parse / interpret / jac (analytic-Jacobian symbolic derivation) / codegen (C++
    source emission) / compile. The Jacobian symbolic work is nested inside
    "generating cpp code", so codegen = that total minus the Jacobian-chain time."""
    b = {"parse": 0.0, "interpret": 0.0, "jac": 0.0, "codegen": 0.0, "compile": 0.0}
    gen = 0.0
    for desc, dur in phases:
        if desc in _PARSE_PHASES:
            b["parse"] += dur
        elif desc.startswith("processing SBML") or desc == "gathering local SBML symbols":
            b["interpret"] += dur
        elif desc == "generating cpp code":
            gen += dur
        elif desc == "compiling cpp code":
            b["compile"] += dur
        elif (desc.startswith("computing ") or desc.startswith("simplifying ")) and _is_jac_func(
            desc.split(" ", 1)[1]
        ):
            b["jac"] += dur
    b["codegen"] = max(gen - b["jac"], 0.0)
    return b


def set_amici_quiet() -> None:
    """Silence AMICI's logger so a sweep's stdout stays readable. The compiler
    subprocess is already muted by ``verbose=False`` in :func:`_build_model`."""
    import logging

    for name in ("amici", "amici.sbml_import", "amici.de_export"):
        logging.getLogger(name).setLevel(logging.ERROR)


def measure_warmup() -> dict:
    """One-time per-process warmup for both engines — call ONCE at worker start,
    before any model, so the heavy imports are charged here (per-process) and not
    to the first model's load. Mirrors ``_rr_common.measure_warmup`` but swaps the
    RoadRunner JIT init for the AMICI import (which pulls SymPy + libSBML + the
    SWIG core). Each job is its own subprocess ⇒ one warmup sample per job.

    - **bngsim:** SymPy import + a trivial ``sp.diff`` (warms the diff machinery so
      the per-model ``last_jacobian_sec`` is pure derivation) — identical to the
      rr_parity measure, keeping the bngsim warmup column comparable across suites.
    - **AMICI:** ``import amici`` + ``import amici.sim.sundials`` wall time.
    """
    t0 = time.perf_counter()
    import sympy as sp
    from sympy.parsing.sympy_parser import parse_expr  # noqa: F401

    _ = sp.diff(sp.Symbol("x") ** 2, sp.Symbol("x"))
    bn_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    import amici  # noqa: F401
    import amici.sim.sundials  # noqa: F401

    amici_sec = time.perf_counter() - t1
    return {
        "bngsim_sec": round(bn_sec, 6),
        "amici_sec": round(amici_sec, 6),
        "amici_source": "import",
    }


# Reverse map AMICI's LinearSolver enum → display name, built lazily on first use
# (the enum lives in amici.sim.sundials, imported per-worker).
_LS_NAMES_CACHE: dict[int, str] | None = None


def _linear_solver_names() -> dict[int, str]:
    global _LS_NAMES_CACHE
    if _LS_NAMES_CACHE is None:
        import amici.sim.sundials as ss

        names = (
            "dense",
            "band",
            "KLU",
            "SuperLUMT",
            "LAPACKDense",
            "LAPACKBand",
            "diag",
            "SPGMR",
            "SPBCG",
            "SPTFQMR",
        )
        out: dict[int, str] = {}
        for n in names:
            v = getattr(ss, f"LinearSolver_{n}", None)
            if v is not None:
                out[int(v)] = n
        _LS_NAMES_CACHE = out
    return _LS_NAMES_CACHE


def _build_model(sbml_str: str):
    """Build-or-load the compiled AMICI model for ``sbml_str``. Returns
    ``(model, timing, cached)``.

    On a cache MISS, sbml2amici runs with ``verbose=logging.DEBUG`` and a phase-
    capture handler so the build decomposes into ``parse_sec`` (libSBML) /
    ``interpret_sec`` (SBML→symbolic) / ``jac_derive_sec`` (analytic-Jacobian
    symbolic derivation) / ``codegen_sec`` (C++ source emission) / ``compile_sec``
    (cmake/ninja/swig/link — the dominant cost). ``load_sec`` (extension load) is
    always paid. On a cache HIT only ``load_sec`` is nonzero. The hash key folds the
    build flags so a flag change never reuses a stale directory."""
    import amici

    # cmake finds SWIG on PATH; make sure the venv's pinned copy is the one it sees.
    ensure_build_path()

    key = hashlib.sha256((sbml_str + _BUILD_FLAGS).encode()).hexdigest()[:16]
    name = f"amici_{key}"
    mdir = AMICI_CACHE / name

    zero = {
        "parse_sec": 0.0,
        "interpret_sec": 0.0,
        "jac_derive_sec": 0.0,
        "codegen_sec": 0.0,
        "compile_sec": 0.0,
    }

    cached = mdir.exists()
    if cached:
        # Load-only fast path. A corrupt cache dir (partial compile) falls through
        # to a fresh build after removal, exactly like bench_ode_vs_amici.py.
        try:
            t0 = time.perf_counter()
            mod = amici.import_model_module(name, str(mdir))
            model = mod.get_model()
            load_sec = time.perf_counter() - t0
            return model, {**zero, "load_sec": round(load_sec, 6)}, True
        except Exception:
            shutil.rmtree(mdir, ignore_errors=True)
            cached = False

    AMICI_CACHE.mkdir(parents=True, exist_ok=True)
    cap = _PhaseCapture()
    base_logger = logging.getLogger("amici")
    base_logger.addHandler(cap)
    saved_levels = {nm: logging.getLogger(nm).level for nm in _AMICI_BUILD_LOGGERS}
    for nm in _AMICI_BUILD_LOGGERS:
        logging.getLogger(nm).setLevel(logging.DEBUG)
    try:
        # verbose=DEBUG opens the phase records; _silence_fds mutes the cmake/ninja
        # subprocess flood AND the logger's console handler (our capture handler
        # writes to a list, not a fd, so it is unaffected).
        with _silence_fds():
            importer = amici.SbmlImporter(sbml_str, from_file=False)
            importer.sbml2amici(
                name,
                str(mdir),
                verbose=logging.DEBUG,
                generate_sensitivity_code=False,
                compute_conservation_laws=False,
                observation_model=[],
            )
        t0 = time.perf_counter()
        mod = amici.import_model_module(name, str(mdir))
        model = mod.get_model()
        load_sec = time.perf_counter() - t0
    except Exception:
        shutil.rmtree(mdir, ignore_errors=True)
        raise
    finally:
        base_logger.removeHandler(cap)
        for nm, lvl in saved_levels.items():
            logging.getLogger(nm).setLevel(lvl)

    b = _bucket_phases(cap.phases)
    return (
        model,
        {
            "parse_sec": round(b["parse"], 6),
            "interpret_sec": round(b["interpret"], 6),
            "jac_derive_sec": round(b["jac"], 6),
            "codegen_sec": round(b["codegen"], 6),
            "compile_sec": round(b["compile"], 6),
            "load_sec": round(load_sec, 6),
        },
        False,
    )


def amici_ode(
    xml: str, t_start: float, t_end: float, n_points: int, rtol: float, atol: float
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """One AMICI CVODES run. Returns ``(time, values[n_time, n_sp], names, timing)``,
    the same signature as ``rr_ode``/``bn_ode`` so the shared comparison path is
    untouched.

    The trajectory is the model's STATE vector ``rdata.x`` (the species AMICI keeps
    as states — with ``compute_conservation_laws=False`` that is the full floating-
    species set), named by ``rdata.state_ids`` (SBML ids). ``align_common`` matches
    these against bngsim by id and compares the intersection — the same partial-
    overlap contract rr_parity uses for RoadRunner's floating-species emission.

    The output grid is ``linspace(t_start, t_end, n_points)`` — identical to the
    grid ``bn_ode`` samples (verified bit-identical in the smoke test), so the
    parity comparison's time-grid check passes.

    A non-success ``rdata.status`` (CVODES failure, e.g. too-much-work) raises so
    the worker classifies it as a reference failure rather than silently comparing
    a truncated trajectory.
    """
    import amici.sim.sundials as ss

    # 1. File I/O (if xml is a path) — isolated from build time, as in rr_ode.
    t0 = time.perf_counter()
    if Path(xml).exists():
        sbml_str = Path(xml).read_text()
        io_sec = time.perf_counter() - t0
    else:
        sbml_str = xml
        io_sec = 0.0

    # 2. Build (parse + codegen/compile) or load from the on-disk cache.
    model, build_timing, cached = _build_model(sbml_str)

    # 3. Solver: matched tolerances, sensitivities OFF (pure forward ODE).
    solver = model.create_solver()
    solver.set_relative_tolerance(rtol)
    solver.set_absolute_tolerance(atol)
    sens_none = getattr(ss, "SensitivityOrder_none", 0)
    solver.set_sensitivity_order(sens_none)
    try:
        ls_code = int(solver.get_linear_solver())
    except Exception:
        ls_code = -1
    linear_solver = _linear_solver_names().get(ls_code, f"kind_{ls_code}")

    # Anchor the initial condition at t_start, matching bn_ode/rr_ode. AMICI's t0
    # defaults to 0 regardless of where the output grid begins, so on a model whose
    # SED-ML initialTime is nonzero it would integrate [0, t_start] FIRST and report
    # an already-evolved state at the grid's first point, while bngsim and
    # RoadRunner both apply the initial state AT t_start. That is a silent
    # apples-to-oranges comparison, not a tolerance effect: it does not shrink as
    # rtol/atol tighten. 21 of the 1323 corpus models have t_start != 0.
    model.set_t0(t_start)
    ts = np.linspace(t_start, t_end, n_points)
    model.set_timepoints(ts)

    # 4. Integration (cold + warm), the shared recipe. AMICI's simulate() is
    # stateless — each call re-integrates from the initial state — so no reset() is
    # needed between reps (contrast bngsim/RR which continue from state). The COLD
    # trajectory feeds the parity verdict; warm reps are best-effort timing only and
    # never change the verdict.
    t1 = time.perf_counter()
    rdata = model.simulate(solver=solver)
    cold_sec = time.perf_counter() - t1
    if int(rdata.status) != 0:
        try:
            msg = ss.simulation_status_to_str(int(rdata.status))
        except Exception:
            msg = str(rdata.status)
        raise RuntimeError(f"AMICI integration failed (status {rdata.status}: {msg})")

    warm: list[float] = []
    for _ in range(_warm_rep_count(cold_sec)):
        try:
            t1 = time.perf_counter()
            rd = model.simulate(solver=solver)
            if int(rd.status) != 0:
                break
            warm.append(time.perf_counter() - t1)
        except Exception:
            break
    integ = _integrate_stats(cold_sec, warm)

    # Full pre-simulation cost decomposition from _build_model's phase capture.
    # parse (libSBML) · interpret (SBML→symbolic) · jac_derive (analytic-Jacobian
    # symbolic derivation, the AMICI analogue of bngsim's jac_derive_sec) · codegen
    # (C++ source emission) · compile (cmake/ninja/swig/link — the dominant cost) ·
    # load (extension import). All 0 except load on a cache hit.
    build_total = (
        build_timing["parse_sec"]
        + build_timing["interpret_sec"]
        + build_timing["jac_derive_sec"]
        + build_timing["codegen_sec"]
        + build_timing["compile_sec"]
        + build_timing["load_sec"]
    )
    timing = {
        "io_sec": round(io_sec, 6),
        "parse_sec": build_timing["parse_sec"],
        "interpret_sec": build_timing["interpret_sec"],
        # jac_derive_sec — AMICI's analytic Jacobian IS derived symbolically (the
        # dxdot/dx chain: dwdx, dxdotdw, dxdotdx_explicit, …); separated here from the
        # rest of codegen so the build tier no longer hides it inside "RHS build".
        "jac_derive_sec": build_timing["jac_derive_sec"],
        # codegen_sec — C++ source EMISSION only (Jacobian symbolic cost removed).
        "codegen_sec": build_timing["codegen_sec"],
        # compile_sec — the cmake-configure + ninja compile + SWIG wrap + static link.
        # This is AMICI's defining, dominant pre-sim cost (a ~25-30s floor); 0 on a
        # cache hit. Kept as its OWN tier, not folded into codegen.
        "compile_sec": build_timing["compile_sec"],
        "load_sec": build_timing["load_sec"],
        # integrate_sec (= warm-min headline) + cold/min/max/median/n_warm.
        **integ,
        # rdata.cpu_time is the pure CVODES integration time in MILLISECONDS — a
        # cross-check on the Python-wall integrate_* numbers (which add SWIG/marshal
        # overhead). Stored, not used for the verdict.
        "integrate_cpu_ms": round(float(rdata.cpu_time), 4),
        # Cross-check / renderer fallback: the full build cost when a consumer wants a
        # single number (parse+interpret+jac+codegen+compile+load).
        "parse_interpret_codegen_sec": round(build_total, 6),
        "config": {
            # AMICI generates per-model C++ and compiles it to a shared library.
            "codegen": "C++ (compiled)",
            # AMICI derives an ANALYTIC (symbolic) Jacobian at codegen — a real
            # contrast with RoadRunner's finite-difference Jacobian.
            "jacobian": "analytical (symbolic)",
            "linear_solver": linear_solver,
            # cached: the compiled extension was reused from the on-disk cache
            # (True), compiled fresh this run (False) — definitive, not inferred.
            "cached": cached,
        },
    }

    names = list(rdata.state_ids)
    x = np.asarray(rdata.x, dtype=float)
    return np.asarray(ts, dtype=float), x, names, timing


def classify_reference_refusal(exc: str) -> str:
    """Map an AMICI failure message to a coarse refusal subclass, mirroring
    ``rr_run.classify_reference_refusal``'s role. Refined as real sweep failures
    surface; the buckets are deliberately broad for v1.

      feature_gap — AMICI cannot import the SBML (unsupported construct, fbc, etc.)
      compile     — C++ generation/compilation failed
      integrator  — CVODES failed at integration time
      other       — anything unclassified
    """
    low = (exc or "").lower()
    if any(
        k in low for k in ("not supported", "unsupported", "fbc", "sbmlexception", "no reactions")
    ):
        return "feature_gap"
    # AMICI compiles per-model C++ by shelling out to ``<model>/setup.py``; a build
    # failure surfaces as a CalledProcessError on that command (the compiler stderr
    # is in the subprocess output, not the truncated exception repr).
    if any(
        k in low
        for k in (
            "compile",
            "gcc",
            "clang",
            "swig",
            "cmake",
            "build failed",
            "calledprocesserror",
            "setup.py",
            "build_ext",
        )
    ):
        return "compile"
    if any(k in low for k in ("integration failed", "cvode", "too_much_work", "status")):
        return "integrator"
    return "other"
