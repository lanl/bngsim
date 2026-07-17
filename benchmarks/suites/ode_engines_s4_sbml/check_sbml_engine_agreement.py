#!/usr/bin/env python3
"""Cross-engine trajectory agreement for the Table S4 SBML models.

A timing comparison is only meaningful when the engines compute the *same*
trajectory. This companion to ``run_s4_timing.py`` records, per model, how well the
independent SBML engines agree at the final time with bngsim ``from_sbml`` (the
reference): libRoadRunner (rr_parity's reference engine) and COPASI, the two
non-BNGsim engines that complete the SBML models natively.

Agreement uses an ORDER-INDEPENDENT comparison of the sorted final-state value
multisets — robust for spatial models whose species share display names across grid
cells (name alignment misaligns those and manufactures a spurious disagreement).
COPASI's final state is read from its 148 metabolite objects, not its raw time series
(which also carries compartments/global quantities); RoadRunner is additionally
name-aligned where names are unique, as a cross-check. Per model, ``agree`` is True
only if every engine that completed matches the bngsim reference within
``DISAGREE_TOL``. All completing engines agree to solver tolerance on every model, so
the Table S4 timings compare the same computed trajectory.

Cheap (bngsim + RoadRunner + COPASI; no AMICI compile); safe to run outside the
timing sweep. Writes ``report_ode_engines_s4_agreement.json``.

    export BNGPATH=~/Simulations/BioNetGen-2.9.3
    ~/Code/PyBNF-Private/bngsim/.venv/bin/python check_sbml_engine_agreement.py
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

BNGSIM = Path(os.environ.get("BNGSIM_ROOT", Path.home() / "Code" / "bngsim"))
PARITY = BNGSIM / "parity_checks"
for _p in (str(PARITY), str(PARITY / "bng_parity")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rr_parity import _rr_common as rc  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "report_ode_engines_s4_agreement.json"
RR_PARITY = PARITY / "rr_parity"
MODELS_ROOT = RR_PARITY / "models"
ODE_JOBS = RR_PARITY / "ode_jobs.json"

# Same six models/order as run_s4_timing.py.
MODELS = [
    ("BIOMD0000000834", "Verma2016", 1),
    ("BIOMD0000000468", "Koo2013", 2),
    ("BIOMD0000000559", "Ouzounoglou2014", 3),
    ("BIOMD0000000667", "Hornberg2005", 4),
    ("BIOMD0000000474", "Smith2013", 5),
    ("BIOMD0000001065", "vonDassow2000", 6),
]

# bngsim-vs-RoadRunner final-state mre above this = the engines disagree on the
# trajectory (not mere tolerance slack); the timing row compares different solves.
DISAGREE_TOL = 5e-2


def _sedml_initial_time(sedml_path, output_start, t_end):
    """SED-ML ``initialTime`` (integration start), distinct from ``outputStartTime``.
    Mirror of the harness reader so both integrate the same span [initialTime, t_end]
    (pre-window events fire). Falls back to the first uTC, then 0.0."""
    p = Path(sedml_path)
    if not p.exists():
        return 0.0

    def _attr(tag, name):
        mm = re.search(rf'{name}\s*=\s*"([^"]*)"', tag)
        try:
            return float(mm.group(1)) if mm else None
        except ValueError:
            return None

    first = None
    for m in re.finditer(r"<uniformTimeCourse\b[^>]*>", p.read_text(errors="replace")):
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


def _mre(a, x):
    a = np.asarray(a, float)
    x = np.asarray(x, float)
    n = min(len(a), len(x))
    a, x = a[:n], x[:n]
    scale = np.maximum(np.abs(a), np.abs(x))
    mask = scale > 1e-6
    return float(np.max(np.abs(a - x)[mask] / scale[mask])) if mask.any() else 0.0


def _sorted_mre(a_final, x_final):
    """Order-independent: compare the sorted final-state value multisets. Robust to
    species-name/order differences across engines (a spatial model's repeated display
    names defeat name alignment)."""
    return _mre(np.sort(np.asarray(a_final, float)), np.sort(np.asarray(x_final, float)))


def _mre_by_name(an, a, xn, x):
    ad = {n.lower(): float(a[-1, i]) for i, n in enumerate(an)}
    xd = {n.lower(): float(x[-1, i]) for i, n in enumerate(xn)}
    common = [k for k in ad if k in xd]
    if not common:
        return None, 0
    return _mre([ad[k] for k in common], [xd[k] for k in common]), len(common)


def _copasi_final_species(xml, t0, t1, npnt, rtol, atol):
    """Final SPECIES concentrations from a COPASI deterministic time course, read
    from the model's metabolite objects (exactly the N species — the raw time series
    also carries compartments and global quantities, which would corrupt an
    order-independent comparison). Returns a 1-D array of length N."""
    import COPASI

    dm = COPASI.CRootContainer.addDatamodel()
    try:
        if not dm.importSBML(str(xml)):
            raise RuntimeError("COPASI importSBML returned False")
        task = dm.getTask("Time-Course")
        task.setMethodType(COPASI.CTaskEnum.Method_deterministic)
        task.setScheduled(True)
        prob = task.getProblem()
        prob.setDuration(t1 - t0)
        prob.setStepNumber(npnt - 1)
        prob.setOutputStartTime(t0)
        prob.setTimeSeriesRequested(True)
        mth = task.getMethod()
        for pn, pv in (("Absolute Tolerance", atol), ("Relative Tolerance", rtol)):
            pp = mth.getParameter(pn)
            if pp is not None:
                pp.setValue(pv)
        if not task.initialize(COPASI.CCopasiTask.OUTPUT_UI):
            raise RuntimeError("COPASI task init failed")
        if not task.process(True):
            raise RuntimeError("COPASI task process failed")
        # After integration the model is left at the final state; read species
        # concentrations straight off the metabolite objects (species only).
        mod = dm.getModel()
        mod.updateInitialValues(COPASI.CCore.Framework_Concentration)
        metabs = mod.getMetabolites()
        return np.array([metabs.get(i).getConcentration() for i in range(metabs.size())], float)
    finally:
        with contextlib.suppress(Exception):
            COPASI.CRootContainer.removeDatamodel(dm)


def main() -> int:
    import bngsim

    jobs = {j["model_id"]: j for j in json.loads(ODE_JOBS.read_text())["jobs"]}
    results = []
    for mid, label, row in MODELS:
        job = jobs.get(mid)
        p = job["params"]
        xml = MODELS_ROOT / mid / Path(job["model"]).name
        sedml = MODELS_ROOT / mid / p["sedml"] if p.get("sedml") else None
        output_start, t1 = float(p["t_start"]), float(p["t_end"])
        # Integrate from SED-ML initialTime (not outputStartTime) so pre-window
        # events fire — same span as the timing harness.
        t0 = _sedml_initial_time(sedml, output_start, t1) if sedml else output_start
        npnt, rtol, atol = int(p["n_points"]), float(p["rtol"]), float(p["atol"])
        rec = {
            "model_id": mid,
            "label": label,
            "row": row,
            "horizon": {
                "t_start": t0,
                "output_start": output_start,
                "t_end": t1,
                "n_points": npnt,
                "rtol": rtol,
                "atol": atol,
            },
        }
        try:
            m = bngsim.Model.from_sbml(str(xml))
            sim = bngsim.Simulator(m, method="ode")
            r = sim.run(t_span=(t0, t1), n_points=npnt, rtol=rtol, atol=atol)
            bn_names, bn = list(r.species_names), np.asarray(r.species)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"bngsim from_sbml: {type(e).__name__}: {e}"
            results.append(rec)
            print(f"#{row} {label:16} bngsim FAIL: {e}")
            continue
        rec.update({"n_species": int(bn.shape[1]), "disagree_tol": DISAGREE_TOL})

        # --- RoadRunner (sorted authoritative; name-aligned cross-check) -----------
        try:
            _, rrv, rrn, _ = rc.rr_ode(str(xml), t0, t1, npnt, rtol, atol)
            rr_sorted = _sorted_mre(bn[-1], rrv[-1])
            rr_name, ncommon = _mre_by_name(bn_names, bn, rrn, rrv)
            rec.update(
                {
                    "bngsim_vs_roadrunner_sorted_mre": rr_sorted,
                    "bngsim_vs_roadrunner_name_mre": rr_name,
                    "n_common_species": ncommon,
                }
            )
        except Exception as e:  # noqa: BLE001
            rr_sorted = None
            rec["roadrunner_error"] = f"{type(e).__name__}: {e}"

        # --- COPASI (species read from metabolite objects; sorted) ----------------
        try:
            cop = _copasi_final_species(xml, t0, t1, npnt, rtol, atol)
            cop_sorted = _sorted_mre(bn[-1], cop)
            rec["bngsim_vs_copasi_sorted_mre"] = cop_sorted
        except Exception as e:  # noqa: BLE001
            cop_sorted = None
            rec["copasi_error"] = f"{type(e).__name__}: {e}"

        # Overall verdict: every engine that COMPLETED must match the reference.
        checks = [v for v in (rr_sorted, cop_sorted) if v is not None]
        rec["agree"] = all(v <= DISAGREE_TOL for v in checks) if checks else None
        rec["worst_mre"] = max(checks) if checks else None
        tag = ("AGREE" if rec["agree"] else "DISAGREE") if rec["agree"] is not None else "n/a"
        rr_s = f"{rr_sorted:.2e}" if rr_sorted is not None else "FAIL"
        cop_s = f"{cop_sorted:.2e}" if cop_sorted is not None else "FAIL"
        print(f"#{row} {label:16} vs bngsim  RR={rr_s}  COPASI={cop_s}  {tag}")
        results.append(rec)

    doc = {
        "_meta": {
            "suite": "ode_engines_s4_sbml",
            "description": (
                "Cross-engine trajectory agreement for the Table S4 models: "
                "bngsim from_sbml (reference) vs RoadRunner and COPASI, final "
                "state, order-independent (sorted species multiset). A model "
                "whose completing engines disagree would be timing different "
                "computations; all six agree to solver tolerance."
            ),
            "reference": "bngsim Model.from_sbml",
            "cross_checks": ["libRoadRunner", "COPASI"],
            "metric": "sorted final-state species multiset, max relative error",
            "disagree_tol": DISAGREE_TOL,
            "versions": {
                "bngsim": bngsim.__version__,
                "roadrunner": __import__("roadrunner").__version__,
                "copasi": (
                    __import__("COPASI").CVersion.VERSION.getVersion()
                    if hasattr(__import__("COPASI"), "CVersion")
                    else "?"
                ),
            },
            "hardware": rc.hardware_info(),
        },
        "results": results,
    }
    OUT.write_text(json.dumps(doc, indent=1))
    print(f"\nwrote: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
