#!/usr/bin/env python3
"""Generate the SSA section of the bngsim-vs-RoadRunner parity test suite.

One "job" per SSA-eligible model (the 355 in ssa_candidates.json -- the subset
that passed validate_for_ssa, minus this-session's manual exclusions). SSA
models are a strict subset of the ODE models, so each SSA job inherits its
time grid from its ODE "sister" in ode_jobs.json (built by build_ode_jobs.py)
rather than re-parsing SED-ML. On top of the inherited horizon we add the
stochastic-only spec that SED-ML doesn't provide:

  * seed + n_rep            -- the replicate ensemble
  * compare = mean_zscore   -- per-(t>0, species) z-score of replicate means
  * effort + max_wall_sec   -- the cost guard. An ODE-appropriate t_end can be
                               a population bomb under SSA, so each job carries
                               its ssa_candidates effort tier and a wall-clock
                               cap; the runner prunes overruns (too_slow).

Tuning note: where a model needs editing to be a sensible SSA test (scale ICs
to bound event count, mark a buffer species boundaryCondition, etc.) the edit
produces a NEW vendored SBML with a unique name. Those are authored during the
run/tune phase; here `derived_from`/`modifications` are null placeholders and
`sbml` points at the original.

Usage:
    python build_ssa_jobs.py [--out ssa_jobs.json]

Output: ssa_jobs.json  ({"_meta":..., "jobs":[...]})
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # so `from _core import ...` resolves
sys.path.insert(0, str(HERE))  # so `import overrides` resolves
import overrides as ov  # noqa: E402
from _core import Job, Oracle, Override, read_manifest, write_manifest  # noqa: E402

SSA_CANDIDATES = Path(
    os.environ.get(
        "SSA_CANDIDATES_JSON",
        HERE / "ssa_candidates.json",  # vendored beside this suite (relocated, GH #108)
    )
)
ODE_JOBS = HERE / "ode_jobs.json"

DEFAULT_SEED = 1
DEFAULT_N_REP = 30
COMPARE_TOL = 5.0  # mean z-score gate (matches parity harness)
WALL_CAP_BY_TIER = {"low": 20.0, "medium": 60.0, "high": 120.0}

# Fallback horizon for SSA models with no ODE sister entry (shouldn't happen
# if both generators ran against the same corpus, but kept honest).
FALLBACK_T_END = 10.0
FALLBACK_N_POINTS = 101


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "ssa_jobs.json")
    args = ap.parse_args()

    if not ODE_JOBS.exists():
        raise SystemExit(f"missing {ODE_JOBS}; run build_ode_jobs.py first.")

    _, ode_jobs = read_manifest(ODE_JOBS)
    ode_by_model = {j.model_id: j for j in ode_jobs}
    candidates = json.loads(SSA_CANDIDATES.read_text())["models"]

    jobs = []
    tier_counts = {"figure_sedml": 0, "template_sedml": 0, "invented": 0}
    effort_counts = {"low": 0, "medium": 0, "high": 0}
    no_ode_sister = []
    # GH #21 detection pass (mirrors build_ode_jobs.py's GH #19 block): SSA jobs
    # whose inherited horizon has outputStartTime (t_start) > initialTime. The old
    # flattening integrated the SSA golden from outputStartTime, dropping pre-window
    # dynamics/events; rr_golden now integrates from initial_time. Recorded in _meta
    # so the affected models are auditable from the committed spec.
    output_start_offset = []

    for m in candidates:
        mid = m["name"]
        effort = m.get("effort", "high")
        ode = ode_by_model.get(mid)
        if ode:
            p = ode.params
            # initial_time = SED-ML initialTime = integration start; inherited from
            # the ODE sister alongside the window (GH #21, mirrors GH #19). .get
            # keeps ODE job files predating the field integrating from t_start.
            initial_time = p.get("initial_time", p["t_start"])
            t_start, t_end, n_points = p["t_start"], p["t_end"], p["n_points"]
            horizon_source = p["horizon_source"]
            sbml_origin, sedml = p["sbml_origin"], p["sedml"]
            model = ode.model
        else:
            no_ode_sister.append(mid)
            initial_time, t_start, t_end, n_points = 0.0, 0.0, FALLBACK_T_END, FALLBACK_N_POINTS
            horizon_source = "invented"
            sbml_origin, sedml = "biomodels_dir", None
            model = f"models/{mid}/{mid}.xml"

        if abs(t_start - initial_time) > 1e-12:
            output_start_offset.append(
                {"model_id": mid, "initial_time": initial_time, "output_start": t_start}
            )

        tier_counts[horizon_source] += 1
        effort_counts[effort] = effort_counts.get(effort, 0) + 1
        jobs.append(
            Job(
                model_id=mid,
                input_format="sbml",
                method="ssa",
                reference_engine="roadrunner",
                model=model,
                oracle=Oracle(metric="mean_zscore", tol=COMPARE_TOL),
                params={
                    "sbml_origin": sbml_origin,
                    "sedml": sedml,
                    "horizon_source": horizon_source,
                    # initial_time = integration start (SED-ML initialTime),
                    # inherited from the ODE sister; the SSA golden integrates
                    # [initial_time, t_end]. t_start = outputStartTime = plotted/
                    # compared window start (>= initial_time). See GH #21 / #19.
                    "initial_time": initial_time,
                    "t_start": t_start,
                    "t_end": t_end,
                    "n_points": n_points,
                    "seed": DEFAULT_SEED,
                    "n_rep": DEFAULT_N_REP,
                    "effort": effort,
                    "max_wall_sec": WALL_CAP_BY_TIER.get(effort, 120.0),
                    # Set when an edited (scaled-IC) variant is vendored.
                    "derived_from": None,
                    "modifications": None,
                },
                overrides=[Override(**o) for o in ov.overrides_for(mid, "ssa")],
            )
        )

    meta = {
        "description": "bngsim-vs-RoadRunner SSA parity jobs; horizon inherited from the ODE sister.",
        "reference_engine": "roadrunner",
        "tier_counts": tier_counts,
        "effort_counts": effort_counts,
        "defaults": {
            "seed": DEFAULT_SEED,
            "n_rep": DEFAULT_N_REP,
            "compare_tol": COMPARE_TOL,
            "wall_cap_by_tier": WALL_CAP_BY_TIER,
        },
        "no_ode_sister": no_ode_sister,
        # GH #21: SSA jobs whose inherited SED-ML outputStartTime (t_start) >
        # initialTime (initial_time). The SSA golden integrates [initial_time,
        # t_end] so pre-window dynamics/events fire; these are the jobs where that
        # matters. Mirrors ode_jobs.json's _meta.output_start_offset (GH #19).
        "output_start_offset": {
            "count": len(output_start_offset),
            "note": (
                "SED-ML outputStartTime > initialTime; the SSA golden integrates "
                "from initial_time, not t_start (GH #21, counterpart of #19)."
            ),
            "models": output_start_offset,
        },
        "notes": (
            "Horizons inherited from ode_jobs.json (SSA is a subset of ODE). The inherited t_end is "
            "the intended window but may be a population bomb under SSA; the runner caps wall time per "
            "effort tier and may prune (too_slow) or call for an edited (scaled-IC) variant. SSA-only "
            "settings (seed/n_rep/tolerance) are authored here -- SED-ML does not carry them. "
            "params.initial_time = integration start (SED-ML initialTime, inherited from the ODE "
            "sister); params.t_start = outputStartTime = plotted/compared window start; the SSA "
            "golden integrates [initial_time, t_end] (GH #21)."
        ),
    }
    stale = ov.stale_keys({j.model_id for j in jobs}, "ssa")
    if stale:
        print(f"WARN: {len(stale)} stale SSA override key(s) match no built job: {stale}")
    n_ov = sum(1 for j in jobs if j.overrides)
    write_manifest(args.out, jobs, meta=meta)
    print(
        f"wrote {args.out} : {len(jobs)} jobs  tiers={tier_counts} effort={effort_counts}  overrides={n_ov}"
        + (f"  no_ode_sister={len(no_ode_sister)}" if no_ode_sister else "")
    )
    if output_start_offset:
        ids = ", ".join(o["model_id"] for o in output_start_offset)
        print(
            f"GH #21: {len(output_start_offset)} SSA job(s) have outputStartTime > initialTime "
            f"(golden integrates from initial_time): {ids}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
