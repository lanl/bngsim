"""Both engines of the AMICI ODE job start at the SED-ML initial time (#1564).

On 19 corpus jobs the SED-ML outputStartTime (``t_start``) is later than the
initialTime (``initial_time``): BIOMD0000000042 on [120, 300], 569 on
[1e-5, 1e-2], 834 on [400, 700]. Each engine takes the start it is handed as the
time the model's initial state holds at, so ``amici_run`` handing both of them
``t_start`` compared two runs of the model's initial state placed at the output
start: nothing before it happened, 834's ``H := 1.8`` at ``t >= 200`` included.
rr_run has integrated from ``initial_time`` since GH #19, and copasi_run since
#1550. The sensitivity job's worker is pinned in ``test_amici_reference_archive``.

These run without AMICI: both engines are stand-ins.
"""

from __future__ import annotations

import json

import _amici_common as ac
import amici_run as ar
import numpy as np


def test_the_initial_time_is_the_jobs_initial_time_or_its_start():
    assert ac.initial_time({"initial_time": 0.0, "t_start": 120.0}) == 0.0
    # A job file predating the field integrates from its t_start, as before.
    assert ac.initial_time({"t_start": 120.0}) == 120.0


def test_both_engines_of_the_ode_job_start_at_the_initial_time(monkeypatch):
    t = np.linspace(0.0, 300.0, 11)
    x = np.column_stack([np.exp(-t / 100.0), 1.0 - np.exp(-t / 100.0)])
    seen = {}

    def engine(name):
        def run(xml, t0, t_end, n_points, rtol, atol, **kw):
            seen[name] = t0
            assert (t_end, n_points) == (300.0, 11)
            return np.linspace(t0, t_end, n_points), x, ["A", "B"], {}

        return run

    monkeypatch.setattr(ac, "measure_warmup", lambda: {})
    monkeypatch.setattr(ac, "set_amici_quiet", lambda: None)
    monkeypatch.setattr(ac, "bn_ode", engine("bngsim"))
    monkeypatch.setattr(ac, "amici_ode", engine("amici"))
    spec = {
        "key": "BIOMD0000000042|ode",
        "model_id": "BIOMD0000000042",
        "method": "ode",
        "metric": "max_rel_err",
        "tol": 1e-4,
        "xml": "unused.xml",
        "params": {
            "initial_time": 0.0,
            "t_start": 120.0,
            "t_end": 300.0,
            "n_points": 11,
            "rtol": 1e-9,
            "atol": 1e-12,
        },
    }
    out: list[dict] = []

    class _Queue:
        put = staticmethod(out.append)

    ar._worker(spec, _Queue())
    (res,) = out
    assert res["status"] == "pass"
    assert seen == {"bngsim": 0.0, "amici": 0.0}


def test_the_curated_manifest_carries_the_source_manifests_times():
    """``amici_ode_jobs.json`` is the runners' default manifest and the nightly
    amici-sens leg's: a curated subset of ``rr_parity/ode_jobs.json`` built before
    the source gained ``initial_time``, so without it two of its jobs
    (BIOMD0000000569, 943) still ran from the outputStartTime. Each curated job's
    params are its source job's."""
    here = ar.HERE
    source = json.loads((ar.RR_ODE_JOBS).read_text())["jobs"]
    curated = json.loads((here / "amici_ode_jobs.json").read_text())["jobs"]
    by_id = {j["model_id"]: j for j in source}
    for job in curated:
        assert job["params"] == by_id[job["model_id"]]["params"], job["model_id"]
        assert "initial_time" in job["params"], job["model_id"]
    offset = {
        j["model_id"] for j in curated if ac.initial_time(j["params"]) != j["params"]["t_start"]
    }
    assert offset == {"BIOMD0000000569", "BIOMD0000000943"}
