"""Per-engine adapters for the SBML semantic test suite.

Each adapter runs one engine and returns a uniform status dict by funnelling its
native output through the shared grading kernel (:mod:`_grading`). The adapter's
only job is the irreducible engine-specific work — load the model, integrate at
the *shared* tolerances, and expose two lookups (``species_conc`` /
``entity_value``). Resolution order, amount/concentration conversion, comparison
tolerances and ``var_missing`` semantics all live in :mod:`_grading`, so every
engine is graded through one and the same path (GH #225).

Adapter contract — each ``run_<engine>(...)`` returns a dict with at least::

    {"status": <pass|load_fail|sim_fail|value_mismatch|var_missing|shape_mismatch|skipped>,
     "error": <str>, "max_err": <float>}

The four engines and how each exposes a value (all concentrations; amounts are
derived uniformly in the kernel as ``conc(t) * vol(t)``):

* **bngsim**       species column (concentration) / observable / parameter,
                   undoing bngsim's internal ``_ant_`` / ``_obs_`` renaming.
* **libRoadRunner** ``[id]`` selection (concentration) / bare ``id`` selection.
* **AMICI**        a ``MeasurementChannel`` observable ``formula=id`` per output
                   variable and per species' compartment (``rdata.y``).
* **COPASI**       time-series ``getConcentrationData`` keyed by SBML id.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
from _grading import (
    SOLVER_ATOL,
    SOLVER_RTOL,
    EngineSeries,
    grade,
)

# On-disk cache of compiled AMICI extensions (AMICI does its own C++ codegen +
# compile, ~25-30 s per model cold). Keyed by SBML content + observable set +
# build flags, so a re-run is load-only. Lives under the git-ignored results/.
_AMICI_CACHE = Path(__file__).resolve().parent / "results" / "amici_cache"


# ── bngsim ───────────────────────────────────────────────────────────────────


class _BngsimSeries:
    """``EngineSeries`` over a bngsim ``Simulator`` result + ``Model``.

    bngsim renames Antimony/SBML ids internally (``_ant_`` for raw ids,
    ``_obs_`` for assignment-rule observables); ``species_conc`` /
    ``entity_value`` undo that so a bare SBML id resolves the same value the
    other engines read directly. Species columns are concentrations (bngsim
    tracks concentrations); the amount conversion is applied uniformly in the
    kernel.
    """

    def __init__(self, model, sim_result, n_points):
        self.model = model
        self.r = sim_result
        self.n_points = n_points

    def _column(self, sbml_id, names_attr, data_attr):
        if self.r is None:
            return None
        names = getattr(self.r, names_attr)
        data = getattr(self.r, data_attr)
        for try_name in (sbml_id, f"_ant_{sbml_id}", f"_obs_{sbml_id}", f"_obs__ant_{sbml_id}"):
            if try_name in names:
                idx = names.index(try_name)
                if idx >= data.shape[1]:
                    # name list longer than the result's columns (e.g. a
                    # single-species model) — treat as absent, not a crash.
                    continue
                return data[:, idx]
        return None

    def _expr_column(self, sbml_id):
        # An AR-target *parameter* whose rule is not a linear sum of species
        # (e.g. ``p2 := 1 + time``) is emitted as a bngsim function/expression,
        # not an observable — read that trajectory column. GH #229.
        if self.r is None:
            return None
        names = self.r.expression_names
        data = self.r.expressions
        for try_name in (sbml_id, f"_ant_{sbml_id}", f"_obs_{sbml_id}", f"_obs__ant_{sbml_id}"):
            if try_name in names:
                return np.asarray(data[try_name])
        return None

    def _rr_species_conc(self, sbml_id):
        if self.r is None:
            return None
        for try_name in (sbml_id, f"_ant_{sbml_id}"):
            try:
                arr = self.r.as_roadrunner([f"[{try_name}]"])
            except ValueError:
                continue
            data = np.asarray(arr)
            if data.ndim == 2 and data.shape[1] == 1:
                return data[:, 0]
        return None

    def species_conc(self, sbml_id):
        # A species' concentration. Prefer BNGsim's RoadRunner-compatible
        # selector because variable-volume hOSU=true species can be stored as
        # amount/V_static while ``[S]`` must report amount/V_live.
        col = self._rr_species_conc(sbml_id)
        if col is not None:
            return col
        # Fallback for older/incomplete results: raw species column, or the
        # observable column bngsim emits for assignment-rule species.
        col = self._column(sbml_id, "species_names", "species")
        if col is not None:
            return col
        return self._column(sbml_id, "observable_names", "observables")

    def entity_value(self, sbml_id):
        # Compartment / parameter / other target: bngsim exposes time-varying
        # compartments and AR targets as observables (and species), constants as
        # parameters. Prefer a trajectory (observable/species/expression column)
        # so a rate-ruled compartment yields vol(t) and a ``p := f(time)`` rule
        # yields p(t); fall back to the constant param.
        col = self._column(sbml_id, "observable_names", "observables")
        if col is not None:
            return col
        col = self._column(sbml_id, "species_names", "species")
        if col is not None:
            return col
        col = self._expr_column(sbml_id)
        if col is not None:
            return col
        for try_name in (sbml_id, f"_ant_{sbml_id}"):
            if try_name in self.model.param_names:
                return np.full(self.n_points, self.model.get_param(try_name))
        return None


class BngsimStageError(Exception):
    """A bngsim load or simulation failure, tagged with which stage failed.

    ``stage`` is the harness status string (``"load_fail"`` / ``"sim_fail"``) so
    both the grading adapter (:func:`run_bngsim`) and the test-runner wrapper
    (GH #241) recover the same distinction from the shared
    :func:`build_bngsim_series` — the adapter maps it to a status dict, the
    wrapper fails closed (emits no CSV)."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def build_bngsim_series(sbml_path, settings):
    """Load an SBML model and integrate it at the shared tolerances.

    Returns ``(_BngsimSeries, n_points)`` — the ``EngineSeries`` the shared
    grader / CSV writer reads. Raises :class:`BngsimStageError` on a load or
    simulation failure (``stage`` records which), leaving the fail-closed policy
    to the caller. This is the single load+integrate path shared by the grading
    adapter and the test-runner wrapper, so both run bngsim identically."""
    import bngsim

    n_points = settings["steps"] + 1
    try:
        model = bngsim.Model.from_sbml(str(sbml_path))
    except Exception as e:
        raise BngsimStageError("load_fail", str(e)) from e

    t_start = settings["start"]
    t_end = t_start + settings["duration"]
    try:
        sim = bngsim.Simulator(model, method="ode")
        sim_result = sim.run(
            t_span=(t_start, t_end),
            n_points=n_points,
            rtol=SOLVER_RTOL,
            atol=SOLVER_ATOL,
        )
    except Exception as e:
        raise BngsimStageError("sim_fail", str(e)) from e

    return _BngsimSeries(model, sim_result, n_points), n_points


def run_bngsim(case_id, sbml_path, settings, exp_data, ent) -> dict:
    try:
        series, _ = build_bngsim_series(sbml_path, settings)
    except BngsimStageError as e:
        return {"status": e.stage, "error": str(e)[:200], "max_err": 0.0}

    return grade(settings, exp_data, series, ent)


# ── libRoadRunner ────────────────────────────────────────────────────────────


class _RoadRunnerSeries:
    """``EngineSeries`` over a RoadRunner result matrix. Species concentration is
    the ``[id]`` column; a compartment / parameter is the bare ``id`` column."""

    def __init__(self, colnames, data):
        self.idx = {cn: i for i, cn in enumerate(colnames)}
        self.data = data

    def species_conc(self, sbml_id):
        i = self.idx.get(f"[{sbml_id}]")
        return self.data[:, i] if i is not None else None

    def entity_value(self, sbml_id):
        i = self.idx.get(sbml_id)
        return self.data[:, i] if i is not None else None


def run_roadrunner(case_id, sbml_path, settings, exp_data, ent) -> dict:
    try:
        import roadrunner
    except ImportError:
        return {"status": "skipped", "error": "roadrunner not installed", "max_err": 0.0}

    try:
        rr = roadrunner.RoadRunner(str(sbml_path))
    except Exception as e:
        return {"status": "load_fail", "error": str(e)[:200], "max_err": 0.0}

    # Build the maximal valid selection: concentration [sp] for every species,
    # bare id for every compartment / global parameter (the amount conversion
    # multiplies by the compartment trajectory in the kernel). Validate each
    # candidate against RoadRunner, which rejects e.g. [compartment].
    fs = list(rr.model.getFloatingSpeciesIds()) + list(rr.model.getBoundarySpeciesIds())
    candidates = ["time"]
    candidates += [f"[{sp}]" for sp in fs]
    candidates += list(rr.model.getCompartmentIds())
    candidates += list(rr.model.getGlobalParameterIds())
    valid = []
    for sel in candidates:
        try:
            rr.selections = valid + [sel]
            valid.append(sel)
        except Exception:
            pass
    rr.selections = valid

    t_start = settings["start"]
    t_end = t_start + settings["duration"]
    n_points = settings["steps"] + 1
    try:
        rr.integrator.absolute_tolerance = SOLVER_ATOL
        rr.integrator.relative_tolerance = SOLVER_RTOL
        result = rr.simulate(t_start, t_end, n_points)
    except Exception as e:
        return {"status": "sim_fail", "error": str(e)[:200], "max_err": 0.0}

    series: EngineSeries = _RoadRunnerSeries(list(result.colnames), np.array(result))
    return grade(settings, exp_data, series, ent)


# ── AMICI ────────────────────────────────────────────────────────────────────


class _AmiciSeries:
    """``EngineSeries`` over an AMICI ``rdata``. Every output variable (and every
    species' compartment) is an observable ``formula=id``; the value is the
    corresponding ``rdata.y`` column. For a non-substance-units species the
    observable is its concentration; the amount conversion is applied in the
    kernel."""

    def __init__(self, obs_ids, y):
        self.idx = {oid: i for i, oid in enumerate(obs_ids)}
        self.y = y

    def _obs(self, sbml_id):
        i = self.idx.get(f"obs_{sbml_id}")
        return self.y[:, i] if i is not None else None

    def species_conc(self, sbml_id):
        return self._obs(sbml_id)

    def entity_value(self, sbml_id):
        return self._obs(sbml_id)


def _amici_build(sbml_path, obs_ids):
    """Build-or-load a compiled AMICI model whose observables are ``obs_{id}`` =
    ``id`` for each id in ``obs_ids``. Returns the loaded model. Cached on disk by
    (SBML content + observable set + build flags)."""
    import amici

    sbml_str = Path(sbml_path).read_text()
    key = hashlib.sha256(
        (sbml_str + "|" + ",".join(sorted(obs_ids)) + "|sens=0;cl=0;sts_v1").encode()
    ).hexdigest()[:16]
    name = f"amici_{key}"
    mdir = _AMICI_CACHE / name

    if mdir.exists():
        try:
            return amici.import_model_module(name, str(mdir)).get_model()
        except Exception:
            shutil.rmtree(mdir, ignore_errors=True)

    _AMICI_CACHE.mkdir(parents=True, exist_ok=True)
    channels = [amici.MeasurementChannel(f"obs_{oid}", formula=oid) for oid in sorted(obs_ids)]
    try:
        importer = amici.SbmlImporter(str(sbml_path))
        importer.sbml2amici(
            name,
            str(mdir),
            observation_model=channels,
            generate_sensitivity_code=False,
            compute_conservation_laws=False,
            verbose=False,
        )
        return amici.import_model_module(name, str(mdir)).get_model()
    except Exception:
        shutil.rmtree(mdir, ignore_errors=True)
        raise


def run_amici(case_id, sbml_path, settings, exp_data, ent) -> dict:
    try:
        import amici  # noqa: F401
    except ImportError:
        return {"status": "skipped", "error": "amici not installed", "max_err": 0.0}

    # Observe every requested variable, plus each species' compartment so the
    # amount conversion has vol(t). Bare-id formulas mirror RR's [id]/id columns.
    obs_ids = set(settings["variables"])
    for v in settings["variables"]:
        comp = ent["species_comp"].get(v)
        if comp:
            obs_ids.add(comp)
    obs_ids = {o for o in obs_ids if o}

    try:
        model = _amici_build(sbml_path, obs_ids)
    except Exception as e:
        return {"status": "load_fail", "error": str(e)[:200], "max_err": 0.0}

    t_start = settings["start"]
    t_end = t_start + settings["duration"]
    n_points = settings["steps"] + 1
    try:
        solver = model.create_solver()
        solver.set_absolute_tolerance(SOLVER_ATOL)
        solver.set_relative_tolerance(SOLVER_RTOL)
        model.set_timepoints(np.linspace(t_start, t_end, n_points))
        rdata = model.simulate(solver=solver)
        if int(rdata.status) != 0:
            return {
                "status": "sim_fail",
                "error": f"AMICI status {rdata.status}",
                "max_err": 0.0,
            }
    except Exception as e:
        return {"status": "sim_fail", "error": str(e)[:200], "max_err": 0.0}

    series: EngineSeries = _AmiciSeries(list(model.get_observable_ids()), np.asarray(rdata.y))
    return grade(settings, exp_data, series, ent)


# ── COPASI ───────────────────────────────────────────────────────────────────


class _CopasiSeries:
    """``EngineSeries`` over a COPASI time series. Both species concentration and
    compartment / parameter values come from ``getConcentrationData`` keyed by
    SBML id (COPASI reports particle numbers via ``getData``; concentration is
    the SBML-comparable quantity, and amounts are derived in the kernel)."""

    def __init__(self, ts, dm, n, const):
        self.ts = ts
        self.n = n
        self.col = {ts.getSBMLId(c, dm): c for c in range(ts.getNumVariables())}
        # SBML id → constant value for entities COPASI keeps OUT of the time
        # series: FIXED/boundary species (initial concentration), constant
        # compartments (volume), and constant global quantities. Without this a
        # constant boundary species (e.g. S1 in 00007) or a constant compartment
        # would read as var_missing for COPASI alone — an accounting asymmetry.
        self.const = const

    def _series(self, sbml_id):
        c = self.col.get(sbml_id)
        if c is None:
            return None
        return np.array([self.ts.getConcentrationData(s, c) for s in range(self.n)])

    def _lookup(self, sbml_id):
        s = self._series(sbml_id)
        if s is not None:
            return s
        if sbml_id in self.const:
            return np.full(self.n, self.const[sbml_id])
        return None

    def species_conc(self, sbml_id):
        return self._lookup(sbml_id)

    def entity_value(self, sbml_id):
        return self._lookup(sbml_id)


def run_copasi(case_id, sbml_path, settings, exp_data, ent) -> dict:
    """COPASI adapter, with process isolation.

    COPASI's native libraries *segfault* on some models (e.g. the ``CSymbolDelay``
    case 00940) — a crash that is NOT a Python exception and would otherwise abort
    the whole multi-engine sweep. So each COPASI case runs in a forked child; a
    crash there is contained and reported as a ``sim_fail`` for that one case,
    keeping the sweep (and the other engines' results) alive. The bngsim / RR /
    AMICI adapters run in-process — they have not been observed to segfault, and
    fork-isolating the subject engine would muddy its timing.
    """
    try:
        import COPASI  # noqa: F401  — load once in the parent so forks inherit it
    except ImportError:
        return {"status": "skipped", "error": "python-copasi not installed", "max_err": 0.0}

    if not hasattr(os, "fork"):  # no fork (Windows): run in-process, unprotected
        return _copasi_core(case_id, sbml_path, settings, exp_data, ent)

    r_fd, w_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child — do the COPASI work, ship the verdict back, never return
        os.close(r_fd)
        try:
            res = _copasi_core(case_id, sbml_path, settings, exp_data, ent)
        except Exception as e:  # noqa: BLE001
            res = {"status": "sim_fail", "error": str(e)[:200], "max_err": 0.0}
        try:
            os.write(w_fd, json.dumps(res).encode())
        finally:
            os.close(w_fd)
            os._exit(0)

    os.close(w_fd)
    chunks = []
    while True:
        b = os.read(r_fd, 65536)
        if not b:
            break
        chunks.append(b)
    os.close(r_fd)
    _, status = os.waitpid(pid, 0)
    if os.WIFSIGNALED(status):
        return {
            "status": "sim_fail",
            "error": f"COPASI crashed in native code (signal {os.WTERMSIG(status)})",
            "max_err": 0.0,
        }
    try:
        return json.loads(b"".join(chunks))
    except Exception:
        return {"status": "sim_fail", "error": "COPASI produced no result", "max_err": 0.0}


def _copasi_core(case_id, sbml_path, settings, exp_data, ent) -> dict:
    import COPASI

    dm = COPASI.CRootContainer.addDatamodel()
    try:
        try:
            if not dm.importSBML(str(sbml_path)):
                return {"status": "load_fail", "error": "COPASI importSBML failed", "max_err": 0.0}
        except Exception as e:
            return {"status": "load_fail", "error": str(e)[:200], "max_err": 0.0}

        task = dm.getTask("Time-Course")
        try:
            task.setMethodType(COPASI.CTaskEnum.Method_deterministic)
            task.setScheduled(True)
            problem = task.getProblem()
            problem.setDuration(settings["duration"])
            problem.setStepNumber(settings["steps"])
            problem.setOutputStartTime(settings["start"])
            problem.setTimeSeriesRequested(True)
            method = task.getMethod()
            for pname, pval in (
                ("Absolute Tolerance", SOLVER_ATOL),
                ("Relative Tolerance", SOLVER_RTOL),
            ):
                p = method.getParameter(pname)
                if p is not None:
                    p.setValue(pval)
            if not task.initialize(COPASI.CCopasiTask.OUTPUT_UI):
                return {"status": "sim_fail", "error": "COPASI task init failed", "max_err": 0.0}
            if not task.process(True):
                return {
                    "status": "sim_fail",
                    "error": "COPASI task process failed",
                    "max_err": 0.0,
                }
        except Exception as e:
            return {"status": "sim_fail", "error": str(e)[:200], "max_err": 0.0}

        # Entities COPASI keeps OUT of the time series (FIXED/boundary species,
        # constant compartments, constant global quantities) — capture their
        # (AR/IA-resolved) initial values by SBML id. Species use initial
        # *concentration* (the SBML-comparable quantity, matching the time-series
        # path); compartments/global quantities use their initial value.
        const = {}
        try:
            cm = dm.getModel()
            for i in range(cm.getNumMetabs()):
                m = cm.getMetabolite(i)
                sid = m.getSBMLId() if hasattr(m, "getSBMLId") else ""
                for key in (sid, m.getObjectName()):
                    if key:
                        const.setdefault(key, m.getInitialConcentration())
            for vec in (cm.getCompartments(), cm.getModelValues()):
                for i in range(vec.size()):
                    obj = vec.get(i)
                    sid = obj.getSBMLId() if hasattr(obj, "getSBMLId") else ""
                    for key in (sid, obj.getObjectName()):
                        if key:
                            const.setdefault(key, obj.getInitialValue())
        except Exception:
            pass

        ts = task.getTimeSeries()
        series: EngineSeries = _CopasiSeries(ts, dm, ts.getRecordedSteps(), const)
        return grade(settings, exp_data, series, ent)
    finally:
        # Free the datamodel so a full 1824-case sweep doesn't accumulate one per
        # case in COPASI's global root container.
        with contextlib.suppress(Exception):
            COPASI.CRootContainer.removeDatamodel(dm)


# Registry: engine key → adapter. ``bngsim`` is the test subject; the rest are
# references. ``run.py`` dispatches the requested set through this map so adding
# an engine is one entry, not a new branch in the orchestrator.
ADAPTERS = {
    "bngsim": run_bngsim,
    "rr": run_roadrunner,
    "amici": run_amici,
    "copasi": run_copasi,
}
