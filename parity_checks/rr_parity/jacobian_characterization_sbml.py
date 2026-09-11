#!/usr/bin/env python3
"""Jacobian characterization of the rr_parity SBML BioModels ODE corpus.

Same metrics and regime classification as
``bng_parity/jacobian_characterization.py`` (structural Jacobian density = nnz/N^2;
stiffness ratio = max|Re lambda| / min-nonzero|Re lambda| over the conservation-reduced
Jacobian, maximized along the trajectory; oscillatory / degenerate handling), but the
model is loaded from SBML via ``Model.from_sbml`` instead of a BNG2.pl-generated ``.net``
-- so there is no network-generation step. The metric helpers are imported from the
bng_parity module so the two corpora are characterized by identical code.

The run is the parity gate's (issue #509): the horizon is the job's — ``initial_time`` to
``t_end`` over ``n_points``, what ``rr_run.py`` integrates — and the tolerances are the
ones the gate forces on both engines, ``_rr_common.DEFAULT_RTOL`` / ``DEFAULT_ATOL`` unless
the job carries a per-model ``tol`` override, never the SED-ML values the gate keeps as
provenance only. Every row records the ``run`` block it used and, when the gate's
``runs/report_ode.json`` is present, the gate's outcome for the model. Trajectory states
are ``Result.state``, the full integrator state, wider than ``Result.species`` on a model
with an event-promoted parameter or compartment (issue #508); a sample whose RHS or
Jacobian is non-finite is skipped and counted rather than voiding the model (issue #510);
and the Jacobian is ``Model.jacobian``'s, differenced when the analytical one is
incomplete, with ``jacobian_method`` saying which (issue #512).

Environment: the bngsim editable checkout venv, e.g.
    ~/Code/bngsim/.venv/bin/python jacobian_characterization_sbml.py --limit 5

Reads ``ode_jobs.json`` + the vendored ``models/<id>/*.xml`` read-only; the only write
is the output report JSON (default ``runs/jacobian_characterization_sbml.json``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BNG_PARITY = HERE.parent / "bng_parity"
for _p in (HERE, BNG_PARITY):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import _rr_common as rc  # noqa: E402  (the gate's shared tolerances)
import jacobian_characterization as jc  # noqa: E402  (shared metric helpers)

INTEGRATION_TIMEOUT = 7.0  # wall-clock cap per model's ODE solve (bounds stiff BioModels;
# BNGsim may retry once with an FD Jacobian, so worst case ~2x)
DEFAULT_REPORT = "runs/report_ode.json"  # the rr gate's report, relative to HERE


def load_ode_jobs() -> list[dict]:
    jobs = json.loads((HERE / "ode_jobs.json").read_text())["jobs"]
    return [j for j in jobs if j.get("method") == "ode"]


def load_gate_outcomes(path: Path | None = None) -> dict[str, str | None]:
    """model_id -> outcome from the rr gate's report, or ``{}`` when there is none.

    The rr gate's horizon and tolerances are reproducible from ``ode_jobs.json`` alone
    (:func:`gate_run_settings`), so its report is not a precondition the way bng_parity's
    is; when present, each row carries the gate's verdict so a characterization can be
    checked against the parity report it describes.
    """
    rp = Path(path) if path is not None else HERE / DEFAULT_REPORT
    if not rp.exists():
        return {}
    rows = json.loads(rp.read_text()).get("results", [])
    return {r["model_id"]: r.get("outcome") for r in rows if r.get("method", "ode") == "ode"}


def gate_tolerances(job: dict) -> tuple[float, float, str]:
    """``(rtol, atol, source)``: the tolerances ``rr_run.py`` forces on both engines for a job.

    The gate ignores the SED-ML ``rtol`` / ``atol`` in the job's params (provenance only)
    and runs every model at the shared ``_rr_common.DEFAULT_RTOL`` / ``DEFAULT_ATOL``,
    unless the job carries a per-model ``tol`` override (an ill-conditioned IVP), which
    wins — the rule and the source of ``rr_run._job_overrides`` (issue #509).
    """
    for o in job.get("overrides") or []:
        if o.get("field") == "tol" and isinstance(o.get("value"), dict):
            return float(o["value"]["rtol"]), float(o["value"]["atol"]), "tol_override"
    return float(rc.DEFAULT_RTOL), float(rc.DEFAULT_ATOL), "gate_default"


def gate_run_settings(job: dict) -> dict:
    """The ``run`` block a row records: the horizon and tolerances the gate ran the job on.

    ``t_start`` is the SED-ML ``initial_time`` (the gate integrates from it so any
    pre-``outputStartTime`` dynamics and events fire, GH #19), falling back to
    ``t_start`` for a job file predating the field. The SED-ML tolerances are kept next
    to the gate's as provenance.
    """
    p = job.get("params") or {}
    rtol, atol, tol_src = gate_tolerances(job)
    return {
        "t_start": float(p.get("initial_time", p.get("t_start", 0.0)) or 0.0),
        "t_end": float(p["t_end"]),
        "n_points": int(p["n_points"]),
        "rtol": rtol,
        "atol": atol,
        "horizon_source": p.get("horizon_source") or "ode_jobs.json",
        "tolerance_source": tol_src,
        "sedml_rtol": p.get("rtol"),
        "sedml_atol": p.get("atol"),
        "integration_timeout_sec": INTEGRATION_TIMEOUT,
    }


def characterize_sbml(model_id: str, sbml_rel: str, job: dict, gate_outcome=None) -> dict:
    """Characterize one SBML model. Never raises: errors -> status field."""
    from bngsim import Model, Simulator

    row: dict = {"model_id": model_id, "status": "ok", "gate_outcome": gate_outcome}
    path = HERE / sbml_rel
    if not path.exists():
        return {**row, "status": "no_sbml"}
    try:
        run = gate_run_settings(job)
        row["run"] = run
        m = Model.from_sbml(str(path))
        core = m._core
        n = int(jc._prop(m, "n_species"))
        if n == 0:
            return {**row, "status": "no_species", "N": 0}
        # The Jacobian the model actually carries (issue #512) and the structural density
        # from it. N is the Jacobian's dimension: the full integrator state, which exceeds
        # the reported species count on a model with an event-promoted parameter or
        # compartment (n_reported_species, recorded after the solve).
        jac_at, rhs_at, jac_fields = jc.jacobian_evaluators(m)
        row.update(jac_fields)
        row["N"] = n
        row["n_reactions"] = int(jc._prop(m, "n_reactions"))
        cl = core.conservation_laws
        row["n_conservation_laws"] = int(cl["n_laws"])
        row["rank"] = n - int(cl["n_laws"])
        pat = jc.density_pattern(jac_at, n)
        row["nnz"] = int(pat.sum())
        row["density"] = row["nnz"] / (n * n)

        if n > jc.EIG_MAX_N:
            return {
                **row,
                "status": "ok_density_only",
                "detail": f"N={n} > EIG_MAX_N; stiffness skipped",
            }
    except Exception as exc:
        import traceback

        return {
            **row,
            "status": "error",
            "detail": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-1200:],
        }

    # Trajectory + stiffness in a SEPARATE try, with a wall-clock integration cap:
    # a pathologically stiff BioModel that CVODE cannot integrate must neither stall
    # the sweep nor lose its already-computed density.
    try:
        res = Simulator(m, method="ode").run(
            t_span=(run["t_start"], run["t_end"]),
            n_points=run["n_points"],
            rtol=run["rtol"],
            atol=run["atol"],
            timeout=INTEGRATION_TIMEOUT,
        )
        row.update(jc.trajectory_fields(res, cl, n, jac_at, rhs_at, pat, run["atol"]))
        return row
    except Exception as exc:
        # keep density (already in row); just no stiffness classification
        row["status"] = "ok_no_stiffness"
        row["detail"] = f"{type(exc).__name__}: {exc}"[:300]
        return row


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out", type=Path, default=HERE / "runs" / "jacobian_characterization_sbml.json"
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument(
        "--report",
        type=Path,
        default=None,
        help="the rr gate's report (rr_run.py's runs/report_ode.json) to take each model's "
        f"gate outcome from (default: {DEFAULT_REPORT} when present)",
    )
    args = ap.parse_args()

    jobs = load_ode_jobs()
    if args.model:
        jobs = [j for j in jobs if args.model in j["model_id"]]
    if args.limit:
        jobs = jobs[: args.limit]
    outcomes = load_gate_outcomes(args.report)

    print(
        f"[jac-sbml] {len(jobs)} SBML ODE models | gate tolerances "
        f"{rc.DEFAULT_RTOL:g}/{rc.DEFAULT_ATOL:g} unless overridden | "
        f"gate outcomes for {sum(1 for j in jobs if j['model_id'] in outcomes)}",
        flush=True,
    )
    rows = []
    t0 = time.perf_counter()
    for k, j in enumerate(jobs, 1):
        r = characterize_sbml(j["model_id"], j["model"], j, outcomes.get(j["model_id"]))
        rows.append(r)
        extra = ""
        if r.get("N") is not None:
            extra = (
                f"N={r['N']} dens={r.get('density', float('nan')):.3f} "
                f"stiff[max/med]={r.get('stiffness_ratio_max', float('nan')):.3g}/"
                f"{r.get('stiffness_ratio_median', float('nan')):.3g} "
                f"npts={r.get('n_time_points', '-')}"
                f"{'/skip' + str(r['n_time_points_skipped']) if r.get('n_time_points_skipped') else ''} "
                f"{r.get('jacobian_method', '-')} "
                f"{'OSC ' if r.get('oscillatory') else ''}"
            )
        print(
            f"[{k:4d}/{len(jobs)}] {str(r.get('status')):16s} {extra}{j['model_id']}", flush=True
        )

    out = {
        "_meta": {
            "generator": "jacobian_characterization_sbml.py",
            "bngsim_version": __import__("bngsim").__version__,
            "n_models": len(rows),
            "tolerances": {
                "rtol": rc.DEFAULT_RTOL,
                "atol": rc.DEFAULT_ATOL,
                "rule": "the gate's shared default on both engines unless the job carries "
                "a per-model tol override (rr_run._job_overrides); the SED-ML values "
                "are recorded per row as sedml_rtol / sedml_atol",
            },
            "gate_report": str(args.report or (HERE / DEFAULT_REPORT)) if outcomes else None,
            "integration_timeout_sec": INTEGRATION_TIMEOUT,
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        },
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    ok = sum(1 for r in rows if str(r.get("status")).startswith("ok"))
    print(
        f"[jac-sbml] wrote {args.out} ({ok}/{len(rows)} characterized, "
        f"{out['_meta']['elapsed_sec']}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
