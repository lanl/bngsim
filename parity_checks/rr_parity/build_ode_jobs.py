#!/usr/bin/env python3
"""Generate the ODE section of the bngsim-vs-RoadRunner parity test suite.

One "job" per ODE-simulable model (the manifest.csv `keep` set). Each job is
a self-contained run spec: which SBML to load, the time grid, solver
tolerances, and the cross-engine comparison oracle. SBML carries no
simulation protocol, so the grid/tolerances come from elsewhere, tiered by
provenance:

  figure_sedml   -- a curated SED-ML that reproduces a published figure
                    (real model-specific horizon). Best tier.
  template_sedml -- an auto-generated `_url.sedml` (generic t_end~10, 4
                    distinct horizons across the whole corpus -- i.e. the
                    flat default we already used). Low added value.
  invented       -- no SED-ML (the uncurated MODEL* ids). Placeholder horizon,
                    flagged for calibrate_horizons.py.

SED-ML source: the local mirror of sys-bio/temp-biomodels `final/<id>/`
(see reference_biomodels_sedml_mirror). For SED-ML-covered models we record
the SBML the SED-ML was authored against (its <model source=...>), since the
SED-ML variable targets resolve against THAT file, not necessarily our
$BIOMODELS_SBML_DIR copy. Vendoring the referenced SBML into the suite is a
separate step.

Usage:
    python build_ode_jobs.py [--mirror DIR] [--out ode_jobs.json]

Output: ode_jobs.json  (list of job dicts under {"_meta":..., "jobs":[...]})
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # so `from _core import ...` resolves
sys.path.insert(0, str(HERE))  # so `import overrides` resolves
import overrides as ov  # noqa: E402
from _core import Job, Oracle, Override, write_manifest  # noqa: E402

# The biomodels corpus-definition artifact (the keep/drop partition) still lives
# under benchmarks/suites/biomodels; override via env if relocated.
SUITE_BIOMODELS = Path(
    os.environ.get(
        "BIOMODELS_SUITE_DIR",
        HERE.parents[1] / "benchmarks" / "suites" / "biomodels",
    )
)
MANIFEST_CSV = SUITE_BIOMODELS / "manifest.csv"
DEFAULT_MIRROR = Path(
    os.environ.get(
        "BIOMODELS_SEDML_DIR",
        os.path.expanduser("~/Code/ssys/biomodels_batch/data/temp-biomodels/final"),
    )
)

# Defaults applied when a tier doesn't supply the value.
DEFAULT_RTOL = 1e-6
DEFAULT_ATOL = 1e-9
INVENTED_T_END = 100.0  # placeholder for the no-SED-ML tier
INVENTED_N_POINTS = 101
COMPARE_TOL = 1e-4  # bngsim-vs-RR max relative error gate (ODE)

# KISAO tolerance parameter ids.
KISAO_RTOL = "KISAO:0000209"
KISAO_ATOL = "KISAO:0000211"


def _lname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_timecourse(path: str) -> dict | None:
    """Extract the first uniformTimeCourse + its model source from a SED-ML file.

    Returns None if the file has no uniformTimeCourse (steady-state / plot-only).

    A SED-ML ``uniformTimeCourse`` carries THREE time fields, which we keep
    distinct (GH #19):

      ``initialTime``     -- where INTEGRATION starts (the model's initial state
                             is applied here). Pre-``outputStartTime`` dynamics and
                             events fire iff we integrate from this time.
      ``outputStartTime`` -- where the plotted/compared OUTPUT window starts;
                             recorded as ``t_start``. May be > ``initialTime``.
      ``outputEndTime``   -- end of both integration and output; recorded as ``t_end``.

    Collapsing ``initialTime`` into ``outputStartTime`` (the old behavior) makes the
    runners integrate ``[outputStartTime, outputEndTime]`` from the initial state,
    skipping everything before ``outputStartTime`` -- e.g. BIOMD834/Verma2016's
    ``H:=1.8 @ t>=200`` event with window ``[400,700]``. We therefore return
    ``initial_time`` separately so the ODE runners can integrate ``[initial_time,
    t_end]``.
    """
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None
    model_source = None
    for el in root.iter():
        if _lname(el.tag) == "model" and model_source is None:
            model_source = el.attrib.get("source")
        if _lname(el.tag) == "uniformTimeCourse":
            a = el.attrib
            steps = a.get("numberOfSteps", a.get("numberOfPoints"))
            rtol = atol = None
            for c in el.iter():
                if _lname(c.tag) == "algorithmParameter":
                    k = c.attrib.get("kisaoID")
                    if k == KISAO_RTOL:
                        rtol = c.attrib.get("value")
                    elif k == KISAO_ATOL:
                        atol = c.attrib.get("value")
            try:
                n_points = int(steps) + 1 if steps is not None else None
            except ValueError:
                n_points = None
            init_t = float(a.get("initialTime", 0.0))
            return {
                # initial_time = integration start; t_start = outputStartTime =
                # plotted/compared window start (>= initial_time). See GH #19.
                "initial_time": init_t,
                "t_start": float(a.get("outputStartTime", a.get("initialTime", 0.0))),
                "t_end": float(a["outputEndTime"]),
                "n_points": n_points,
                "rtol": float(rtol) if rtol else None,
                "atol": float(atol) if atol else None,
                "model_source": model_source,
            }
    return None


def best_sedml(model_dir: Path) -> tuple[str | None, str]:
    """Pick one SED-ML for a model dir. Prefer a figure (non-_url) file with a
    real time-course; fall back to the _url template. Returns (path, tier)."""
    sedmls = sorted(glob.glob(str(model_dir / "*.sedml")))
    figure = [p for p in sedmls if not p.endswith("_url.sedml")]
    template = [p for p in sedmls if p.endswith("_url.sedml")]
    for p in figure:
        if parse_timecourse(p):
            return p, "figure_sedml"
    for p in template:
        if parse_timecourse(p):
            return p, "template_sedml"
    return None, "invented"


def keep_models() -> list[str]:
    out = []
    with open(MANIFEST_CSV) as f:
        next(f)  # leading comment line
        for row in csv.DictReader(f):
            if row["verdict"] == "keep":
                out.append(row["model_id"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mirror", type=Path, default=DEFAULT_MIRROR)
    ap.add_argument("--out", type=Path, default=HERE / "ode_jobs.json")
    args = ap.parse_args()

    models = keep_models()
    jobs = []
    tier_counts = {"figure_sedml": 0, "template_sedml": 0, "invented": 0}
    # GH #19 detection pass: SED-ML horizons whose outputStartTime > initialTime.
    # The old flattening integrated from outputStartTime, dropping pre-window
    # dynamics/events; the ODE runners now integrate from initial_time. Recorded
    # in _meta so the affected models are auditable from the committed spec.
    output_start_offset = []

    for mid in models:
        mdir = args.mirror / mid
        path, tier = best_sedml(mdir) if mdir.is_dir() else (None, "invented")
        tc = parse_timecourse(path) if path else None

        if tc:
            sbml = tc["model_source"] or f"{mid}.xml"
            initial_time = tc["initial_time"]
            t_start, t_end = tc["t_start"], tc["t_end"]
            n_points = tc["n_points"] or INVENTED_N_POINTS
            rtol = tc["rtol"] or DEFAULT_RTOL
            atol = tc["atol"] or DEFAULT_ATOL
            sbml_origin = "temp-biomodels"
        else:
            sbml = f"{mid}.xml"
            initial_time, t_start, t_end, n_points = 0.0, 0.0, INVENTED_T_END, INVENTED_N_POINTS
            rtol, atol = DEFAULT_RTOL, DEFAULT_ATOL
            sbml_origin = "biomodels_dir"

        if abs(t_start - initial_time) > 1e-12:
            output_start_offset.append(
                {"model_id": mid, "initial_time": initial_time, "output_start": t_start}
            )

        sedml = os.path.basename(path) if path else None
        tier_counts[tier] += 1
        jobs.append(
            Job(
                model_id=mid,
                input_format="sbml",
                method="ode",
                reference_engine="roadrunner",
                # Vendored model dir holds the SBML (gitignored) + SED-ML beside it.
                model=f"models/{mid}/{sbml}",
                oracle=Oracle(metric="max_rel_err", tol=COMPARE_TOL),
                params={
                    "sbml_origin": sbml_origin,
                    "sedml": sedml,
                    "horizon_source": tier,
                    # initial_time = integration start (SED-ML initialTime); the
                    # runners integrate [initial_time, t_end]. t_start = SED-ML
                    # outputStartTime = plotted/compared window start (GH #19).
                    "initial_time": initial_time,
                    "t_start": t_start,
                    "t_end": t_end,
                    "n_points": n_points,
                    "rtol": rtol,
                    "atol": atol,
                },
                overrides=[Override(**o) for o in ov.overrides_for(mid, "ode")],
                notes="invented horizon (placeholder)" if tier == "invented" else "",
            )
        )

    meta = {
        "description": "bngsim-vs-RoadRunner ODE parity jobs; horizons tiered by SED-ML provenance.",
        "reference_engine": "roadrunner",
        # Symbolic, not the resolved path: serializing str(args.mirror) would
        # bake a machine-specific /Users/<name>/... into the committed spec.
        "sedml_source": "$BIOMODELS_SEDML_DIR (sys-bio/temp-biomodels final/<id>/, pinned @6a09daf4; see temp_biomodels_manifest.json)",
        "tier_counts": tier_counts,
        "defaults": {
            "rtol": DEFAULT_RTOL,
            "atol": DEFAULT_ATOL,
            "invented_t_end": INVENTED_T_END,
            "compare_tol": COMPARE_TOL,
        },
        # GH #19: models whose SED-ML outputStartTime (t_start) > initialTime
        # (initial_time). The ODE runners integrate [initial_time, t_end] so
        # pre-window dynamics/events fire; these are the jobs where that matters.
        "output_start_offset": {
            "count": len(output_start_offset),
            "note": (
                "SED-ML outputStartTime > initialTime; runners integrate from "
                "initial_time, not t_start (GH #19)."
            ),
            "models": output_start_offset,
        },
        "notes": (
            "model = models/<model_id>/<sbml>; materialize.py places the SBML (gitignored) and "
            "SED-ML there. invented-tier horizons are placeholders for calibrate_horizons.py. "
            "params.initial_time = integration start (SED-ML initialTime); params.t_start = "
            "outputStartTime = plotted/compared window start; the ODE runners integrate "
            "[initial_time, t_end] (GH #19)."
        ),
    }
    stale = ov.stale_keys({j.model_id for j in jobs}, "ode")
    if stale:
        print(f"WARN: {len(stale)} stale ODE override key(s) match no built job: {stale}")
    n_ov = sum(1 for j in jobs if j.overrides)
    write_manifest(args.out, jobs, meta=meta)
    print(f"wrote {args.out} : {len(jobs)} jobs  tiers={tier_counts}  overrides={n_ov}")
    if output_start_offset:
        ids = ", ".join(o["model_id"] for o in output_start_offset)
        print(
            f"GH #19: {len(output_start_offset)} job(s) have outputStartTime > initialTime "
            f"(runners integrate from initial_time): {ids}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
