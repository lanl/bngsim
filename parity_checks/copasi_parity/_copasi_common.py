"""Shared machinery for the copasi_parity suite (COPASI reference adapter).

This is the COPASI analogue of ``rr_parity/_rr_common.py``'s reference block
(``rr_ode``) and ``amici_parity/_amici_common.py``'s (``amici_ode``). The bngsim
test side (``bn_ode``) and every engine-agnostic helper (``align_common``,
``_integrate_stats``, ``_warm_rep_count``, ``schedule``, ``hardware_info``,
``load_and_filter``, ``model_path``, ``sundials_version``,
``DEFAULT_RTOL``/``DEFAULT_ATOL``) are imported verbatim from
``rr_parity._rr_common`` so the three suites share ONE bngsim adapter and ONE
comparison path — only the *reference* engine changes.

Public surface:
  * ``co_ode``               — one COPASI deterministic timecourse, same
                               signature/return as ``rr_ode``/``amici_ode``.
  * ``sbml_core_reading``    — the SBML with distribution annotations taken off
                               function definitions, the model the other engines
                               read (issue #1563).
  * ``measure_warmup``       — per-process warmup (bngsim SymPy + COPASI import).
  * ``set_copasi_quiet``     — drain COPASI's message deque between models.
  * ``classify_reference_refusal`` — map a COPASI failure to a refusal subclass.

COPASI has no per-model codegen/compile step (contrast AMICI), so a sweep is as
cheap as rr_parity: import (libSBML parse + COPASI model build) + LSODA solve.
The deterministic timecourse uses COPASI's default LSODA integrator.
"""

from __future__ import annotations

import contextlib
import locale
import time
from pathlib import Path

import numpy as np


@contextlib.contextmanager
def keep_locale():
    """Restore the process ``LC_CTYPE`` after the block, whatever COPASI did to it.

    COPASI sets ``LC_CTYPE`` to ``C`` when its module loads, and again when it imports
    SBML whose function definitions carry an annotation (issue #1563: 589's RUNIFORM,
    or any annotation there). Every COPASI import goes through this, for the reason
    ``_import_copasi`` gives."""
    saved = locale.setlocale(locale.LC_CTYPE)
    try:
        yield
    finally:
        with contextlib.suppress(Exception):
            locale.setlocale(locale.LC_CTYPE, saved)


def _import_copasi():
    """Import COPASI while preserving the process ``LC_CTYPE``.

    COPASI's SWIG/C++ module init calls ``setlocale`` and can drop ``LC_CTYPE`` to
    ``C`` (ASCII). That is invisible to COPASI itself but silently breaks the *test*
    engine: bngsim's ``bn_ode`` reads each SBML with ``Path.read_text()``, whose
    default encoding follows ``locale.getpreferredencoding()`` — so after COPASI
    loads, a UTF-8 SBML (accented author names, µ, …) raises ``UnicodeDecodeError``
    in bngsim, misattributing a reference-engine side effect as a bngsim bug. Saving
    and restoring around the (one-time) import keeps the reference engine from
    corrupting the test engine. Subsequent imports are cached no-ops."""
    with keep_locale():
        import COPASI

        return COPASI


# Reuse the bngsim adapter + engine-agnostic helpers from rr_parity. _rr_common
# resolves on sys.path because copasi_run.py inserts parity_checks/ (its parent)
# before importing this module — the same bootstrap amici_parity uses.
from rr_parity import _rr_common as _rc  # noqa: E402

bn_ode = _rc.bn_ode
align_common = _rc.align_common
schedule = _rc.schedule
hardware_info = _rc.hardware_info
load_and_filter = _rc.load_and_filter
model_path = _rc.model_path
sundials_version = _rc.sundials_version
_integrate_stats = _rc._integrate_stats
_warm_rep_count = _rc._warm_rep_count

DEFAULT_RTOL = _rc.DEFAULT_RTOL
DEFAULT_ATOL = _rc.DEFAULT_ATOL


def copasi_version() -> str | None:
    """The COPASI build version (e.g. ``4.46.300``), for the report's version stamp."""
    try:
        return _import_copasi().__version__
    except Exception:
        return None


def set_copasi_quiet() -> None:
    """Drain COPASI's global message deque so a long sweep does not accumulate
    per-model warnings (and so ``getAllMessageText`` on a failure returns only that
    model's messages). Best-effort; COPASI's logger is otherwise silent by default.
    """
    COPASI = _import_copasi()

    with contextlib.suppress(Exception):
        COPASI.CCopasiMessage.clearDeque()


def measure_warmup() -> dict:
    """One-time per-process warmup for both engines — call ONCE at worker start,
    before any model, so the heavy imports are charged here (per-process) and not
    to the first model's load. Mirrors ``_rr_common.measure_warmup`` but swaps the
    RoadRunner JIT init for the COPASI import (libSBML + the SWIG core). Each job is
    its own subprocess ⇒ one warmup sample per job.

    - **bngsim:** SymPy import + a trivial ``sp.diff`` (warms the diff machinery so
      the per-model ``last_jacobian_sec`` is pure derivation) — identical to the
      rr_parity / amici_parity measure, keeping the bngsim warmup column comparable.
    - **COPASI:** ``import COPASI`` wall time (dlopen of the SWIG extension).
    """
    t0 = time.perf_counter()
    import sympy as sp

    _ = sp.diff(sp.Symbol("x") ** 2, sp.Symbol("x"))
    bn_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    _import_copasi()

    co_sec = time.perf_counter() - t1
    return {
        "bngsim_sec": round(bn_sec, 6),
        "copasi_sec": round(co_sec, 6),
        "copasi_source": "import",
    }


def _copasi_msg() -> str:
    """The current COPASI message text (its errors surface here, not as a raised
    exception), truncated for the report."""
    COPASI = _import_copasi()

    try:
        return COPASI.CCopasiMessage.getAllMessageText().strip()[:300]
    except Exception:
        return ""


def sbml_source(xml: str) -> tuple[str | None, str | None]:
    """``(path, None)`` for a file COPASI should read itself, ``(None, text)`` for
    SBML text.

    Text is recognized first, as ``rr_ode`` and ``amici_ode`` do: SBML text can be
    longer than a file name may be, and ``Path.exists()`` raises on it (``File
    name too long``) rather than answering False (issue #1550).
    """
    if not str(xml).lstrip().startswith("<") and Path(xml).exists():
        return str(Path(xml)), None
    return None, xml


# The SBML "distribution" annotation (the scheme before the distrib package). A
# function definition carrying it, such as BIOMD0000000589's
#   RUNIFORM(a, b) = (a + b) / 2  with  <distribution definition=".../uniform"/>,
# has the mean as its SBML math and a distribution as its annotation. bngsim,
# RoadRunner and AMICI evaluate the math. COPASI reads the annotation and draws
# a fresh value on every run, so its reference is a different model each time.
DISTRIBUTION_ANNOTATION_NS = "http://sbml.org/annotations/distribution"


def sbml_core_reading(text: str) -> tuple[str, list[str]]:
    """``(sbml, ids)``: the SBML with the distribution annotation removed from each
    function definition that carries one, and those definitions' ids.

    Without it COPASI samples such a function instead of evaluating its SBML math,
    which every other engine here evaluates (issue #1563). Only that annotation
    element goes; the rest of each annotation, and the model, are written back as
    libsbml read them. Text without the annotation comes back unchanged, unparsed.
    """
    if DISTRIBUTION_ANNOTATION_NS not in text:
        return text, []
    import libsbml

    doc = libsbml.readSBMLFromString(text)
    model = doc.getModel()
    dropped: list[str] = []
    for f in model.getListOfFunctionDefinitions() if model is not None else []:
        if f.isSetAnnotation() and DISTRIBUTION_ANNOTATION_NS in f.getAnnotationString():
            status = f.removeTopLevelAnnotationElement("distribution", DISTRIBUTION_ANNOTATION_NS)
            if status != libsbml.LIBSBML_OPERATION_SUCCESS:
                raise RuntimeError(
                    f"could not remove the distribution annotation from {f.getId()}"
                )
            dropped.append(f.getId())
    if not dropped:
        return text, []
    return libsbml.writeSBMLToString(doc), dropped


def co_ode(
    xml: str, t_start: float, t_end: float, n_points: int, rtol: float, atol: float
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """One COPASI deterministic timecourse. Returns ``(time, values[n_time, n_sp],
    names, timing)`` — the same signature as ``rr_ode``/``amici_ode`` so the shared
    comparison path is untouched.

    The trajectory is the SPECIES set only: each time-series variable is kept iff
    its SBML id matches a model metabolite (COPASI's raw series also carries Time,
    compartments, and global quantities, which would corrupt the id-aligned
    comparison). Values are **concentrations** (``getConcentrationData``), named by
    SBML id so ``align_common`` matches them against bngsim by id — the same
    partial-overlap contract rr_parity uses for RoadRunner's floating species.

    The output grid is COPASI's own recorded time column: it integrates over
    ``[model_t0, t_end]`` and records ``n_points`` rows on ``linspace(t_start,
    t_end, n_points)`` from ``t_start`` on, when that lands on COPASI's grid
    anchored at ``model_t0`` (see step 3 for when it does not). Returning the real
    column (not a synthetic ``linspace``) keeps the differ's time-grid check honest
    if a model with events forces a different grid.

    COPASI integrates from the model's own initial time whatever ``t_start`` is,
    and ``t_start`` only says where recording begins. ``bn_ode`` instead starts
    the model's initial state at its ``t_start``. So the two solve the same problem
    only when the caller passes the job's initial time, as ``copasi_run`` does
    (issue #1550): given an output start past it, bngsim's run was the same model
    shifted in time by the gap.

    COPASI is given ``sbml_core_reading`` of the model, so a function definition
    with a distribution annotation is evaluated, as the other engines evaluate
    it, rather than sampled (issue #1563). ``timing["distribution_annotations_dropped"]``
    names the definitions that applies to, empty for every other model.

    A failed ``importSBML`` (unsupported construct) or a failed ``process`` (LSODA
    divergence) raises, so the worker classifies it as a reference failure rather
    than silently comparing a truncated trajectory.
    """
    COPASI = _import_copasi()

    set_copasi_quiet()

    # 0. Outside the timed phases: the SBML-core reading of a function definition
    #    COPASI would otherwise sample (issue #1563). A model without one keeps
    #    its path, and COPASI reads the file itself as before.
    sbml_path, sbml_str = sbml_source(xml)
    text = sbml_str if sbml_str is not None else Path(sbml_path).read_text(encoding="utf-8")
    core, dropped = sbml_core_reading(text)
    if dropped:
        sbml_path, sbml_str = None, core

    # 1. Resolve the SBML source: a path (the manifest passes one) or a string.
    t0 = time.perf_counter()
    io_sec = 0.0 if sbml_path is not None else time.perf_counter() - t0

    dm = COPASI.CRootContainer.addDatamodel()
    try:
        # 2. Import (libSBML parse + COPASI model build). importSBML returns False
        #    on an unsupported construct; the reason is in the message deque.
        t1 = time.perf_counter()
        with keep_locale():
            ok = (
                dm.importSBML(sbml_path)
                if sbml_path is not None
                else dm.importSBMLFromString(sbml_str)
            )
        if not ok:
            raise RuntimeError(f"COPASI importSBML failed: {_copasi_msg()}")
        load_sec = time.perf_counter() - t1

        mod = dm.getModel()

        # 3. Configure the deterministic (LSODA) timecourse to reproduce the shared
        # output grid linspace(t_start, t_end, n_points). ``t_start`` is where
        # recording begins, NOT the model's initial time: COPASI always integrates
        # from ``model_t0`` and only begins *recording* at ``t_start``. copasi_run
        # passes the job's initial time, so the corpus records from ``model_t0``
        # (see the docstring). COPASI's recorded grid is uniform and anchored at
        # ``model_t0``, so it reproduces the target grid EXACTLY only when ``t_start``
        # lands on that anchored grid — i.e. (t_start - model_t0)/step_size is
        # integral (the common t_start==0 case, and windows like [120,300] step 0.3).
        # When it does not (e.g. [100,400] step 0.3, or [1e-5,1e-2]), COPASI's nearest
        # grid point is offset by a fraction of a step; we then integrate on a
        # REFINE-times finer grid and cubic-spline the trajectory onto the exact
        # target grid. The spline is knot-exact where the grids coincide and
        # sub-tolerance elsewhere, so it removes the spurious time-grid mismatch
        # without ever manufacturing a divergence.
        model_t0 = mod.getInitialTime()
        step_size = (t_end - t_start) / max(n_points - 1, 1)
        target = np.linspace(t_start, t_end, n_points)
        frac = (t_start - model_t0) / step_size if step_size else 0.0
        aligned = abs(frac - round(frac)) < 1e-6

        task = dm.getTask("Time-Course")
        task.setMethodType(COPASI.CTaskEnum.Method_deterministic)
        task.setScheduled(True)
        prob = task.getProblem()
        prob.setTimeSeriesRequested(True)
        prob.setAutomaticStepSize(False)
        prob.setDuration(t_end - model_t0)
        if aligned:
            prob.setStepNumber(int(round((t_end - model_t0) / step_size)))
            prob.setOutputStartTime(t_start)
        else:
            refine = 8
            prob.setStepNumber(int(round((t_end - model_t0) / (step_size / refine))))
            prob.setOutputStartTime(model_t0)  # record the full span; interpolate below
        mth = task.getMethod()
        for pname, pval in (("Absolute Tolerance", atol), ("Relative Tolerance", rtol)):
            pp = mth.getParameter(pname)
            if pp is not None:
                pp.setValue(float(pval))
        if not task.initialize(COPASI.CCopasiTask.OUTPUT_UI):
            raise RuntimeError(f"COPASI task init failed: {_copasi_msg()}")

        # 4a. COLD solve — its trajectory feeds the parity verdict.
        t2 = time.perf_counter()
        if not task.process(True):
            raise RuntimeError(f"COPASI process failed: {_copasi_msg()}")
        cold_sec = time.perf_counter() - t2

        # Extract the cold trajectory into numpy NOW, before any warm re-process
        # can overwrite the task's time series. Keep every dynamic quantity COPASI
        # reports, keyed by SBML id: species AND rate-rule/ODE-driven parameters or
        # compartments that reaction-free models expose as states (Hodgkin-Huxley
        # V, m, h, n and FitzHugh-Nagumo v, u live as SBML *parameters*, not
        # metabolites). ``align_common`` intersects these with bngsim's state ids, so
        # COPASI columns bngsim does not report are dropped — matching ``rr_ode``,
        # which likewise returns rate-rule target parameters, not only species. Var 0
        # ("Time") is excluded; a variable with no SBML id cannot align and is skipped.
        ts = task.getTimeSeries()
        n_rec = ts.getRecordedSteps()
        n_var = ts.getNumVariables()
        state_cols = [v for v in range(n_var) if ts.getTitle(v) != "Time" and ts.getSBMLId(v, dm)]
        names = [ts.getSBMLId(v, dm) for v in state_cols]
        raw_t = np.array([ts.getConcentrationData(s, 0) for s in range(n_rec)], dtype=float)
        raw_v = np.array(
            [[ts.getConcentrationData(s, v) for v in state_cols] for s in range(n_rec)],
            dtype=float,
        )
        if aligned or raw_t.shape[0] < 2 or not state_cols:
            time_arr, values = raw_t, raw_v
        else:
            from scipy.interpolate import CubicSpline

            # target ⊆ [model_t0, t_end] = [raw_t[0], raw_t[-1]] ⇒ no extrapolation.
            values = CubicSpline(raw_t, raw_v, axis=0)(target)
            time_arr = target

        # 4b. WARM reps — best-effort timing only; never let a re-solve failure
        #     change the verdict (mirrors rr_ode / amici_ode).
        warm: list[float] = []
        for _ in range(_warm_rep_count(cold_sec)):
            try:
                mod.applyInitialValues()
                if not task.initialize(COPASI.CCopasiTask.OUTPUT_UI):
                    break
                t3 = time.perf_counter()
                if not task.process(True):
                    break
                warm.append(time.perf_counter() - t3)
            except Exception:
                break
        integ = _integrate_stats(cold_sec, warm)

        timing = {
            "io_sec": round(io_sec, 6),
            "load_sec": round(load_sec, 6),
            "distribution_annotations_dropped": dropped,
            **integ,
            "config": {
                "integrator": "LSODA (deterministic)",
                "jacobian": "internal (LSODA)",
                "linear_solver": "dense (LSODA)",
            },
        }
        return time_arr, values, names, timing
    finally:
        with contextlib.suppress(Exception):
            COPASI.CRootContainer.removeDatamodel(dm)


def classify_reference_refusal(exc: str) -> str:
    """Map a COPASI failure message to a coarse refusal subclass, mirroring
    ``amici_parity.classify_reference_refusal``'s role. COPASI has no per-model
    compile step, so there is no ``compile`` bucket.

      feature_gap — COPASI cannot import the SBML (unsupported construct, fbc, …)
      integrator  — LSODA failed at integration time
      other       — anything unclassified
    """
    low = (exc or "").lower()
    if any(
        k in low
        for k in ("importsbml", "not supported", "unsupported", "fbc", "could not", "sbml (")
    ):
        return "feature_gap"
    if any(k in low for k in ("process failed", "task init", "lsoda", "integration", "step size")):
        return "integrator"
    return "other"
